# This script tries to compute the variance of the direct importance sampling for Gaussian mixtures.

push!(LOAD_PATH,"../src")
using ModPotential
using LinearAlgebra
using Printf

##################
## 2D example
##################

function test2(λ)
    # test 2
    #λ = 2.0
    #λ = 5.0
    n = 2
    σ = sqrt(0.1)
    μlist = [[λ, 0.0], [0.0, -λ]]
    Σlist = [σ^2*Matrix(1.0I, n, n) for i = 1:2]
    ωlist = [0.2, 0.8]
    
    print_result(ωlist, μlist, Σlist)
end

@printf("\n2D Example\n")
test2(2.0)
test2(5.0)

##################
## High-D example
##################

function testhigh(n, λ, σsq₁, σsq₂)
    num_pts = 4
    μlist = []
    Σlist = []

    mode = num_pts
    weight = ones(mode)*(1/σsq₁/σsq₂^(n/2-1))/mode

    for i = 1:num_pts
        θ = i*(2*pi/num_pts)
        push!(μlist, vcat(λ*[cos(θ), sin(θ)], zeros(n-2)))
        sigma_diag = vcat([σsq₁, σsq₁], σsq₂*ones(n-2))
        push!(Σlist, Diagonal(sigma_diag))
    end
    
    print_result(weight, μlist, Σlist)
end

@printf("\nHigh-D Example\n")
#n = 20 
σsq₁ = 0.1 
σsq₂ = 0.5
testhigh(2, 5.0, σsq₁, σsq₂)
testhigh(10, 5.0, σsq₁, σsq₂)

##################
## Mix 25
##################

function test_mix25(n, σsq₁, σsq₂)
    num_pts = 25
    μlist = []
    Σlist = []

    mode = num_pts
    weight = ones(mode)*(1/σsq₁/σsq₂^(n/2-1))/mode

    for i = 1:num_pts
        for j = 1:num_pts
            push!(μlist, vcat([i/sqrt(5), j/sqrt(5)], zeros(n-2)))
            sigma_diag = vcat([σsq₁, σsq₁], σsq₂*ones(n-2))
            push!(Σlist, Diagonal(sigma_diag))
        end
    end
    
    print_result(weight, μlist, Σlist)
end

@printf("\nMix 25\n")
test_mix25(20, 0.01/5, 0.1/5)
test_mix25(40, 0.01/5, 0.1/5)
