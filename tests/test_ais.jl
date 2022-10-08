push!(LOAD_PATH,"../src")
using LinearAlgebra
using Statistics
using Plots
using Plots.PlotMeasures
using StatsPlots
using Printf
using NEIS
using Test

# set up the model
n = 2
λ = 2.0
σsq = 0.5
U₀, U₁, exact_mean = load_eg1(λ; σsq=σsq);

chain_length = 10^6
τ = 0.1
@time state, decision = MALA_OLD_chain(n, U₁, τ, chain_length, randn(n));

@printf("acceptance rate %.2E\n", sum(decision)/chain_length);

sdx = [item[1] for item in state]
sdy = [item[2] for item in state]

figsize = (300,200)
fig = Plots.histogram2d(sdx, sdy, size=figsize, fill=true, color=:tofino,
    left_margin=20px, right_margin=20px, normed=true, nbinsx=30, nbinsy=30)

@testset "test ais and direct importance sampling" begin
    # test AIS
    reset_query_stat(U₁)
    K = 10
    numsample = 10^5
    βlist = Array(range(0, stop=1.0, length=K+1))
    @time ais_estimate = ais_neal(U₀, U₁, K, numsample)[2]
    @test abs(ais_estimate/exact_mean-1.0) < 0.05
    print_query_stat(U₁)
    stat = get_query_stat(U₁)
    @test norm(stat[1] - 2*numsample*K) < 1.0e-5
    @test norm(stat[2] - 2*numsample*K) < 1.0e-5

    # test Vanilla Importance Sampling
    @time vanilla_estimate = vanilla_importance_sampling(U₀, U₁, numsample*100)[2]
    @test abs(vanilla_estimate/exact_mean-1.0) < 0.05
end

fig
