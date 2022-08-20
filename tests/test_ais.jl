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
    K = 10
    numsample = 10^5
    βlist = Array(range(0, stop=1.0, length=K+1))
    @time data = map(j->ais_neal(sampler(U₀,1)[:], n, U₀, U₁, K, βlist, τ)[1], 1:numsample);
    @printf "error %.2E\n" abs(mean(data)/exact_mean-1.0)
    @test abs(mean(data)/exact_mean-1.0) < 0.05

    # test Vanilla Importance Sampling
    @time _, vanilla_m, vanilla_var = vanilla_importance_sampling(U₀, U₁, numsample*100);
    @test abs(vanilla_m/exact_mean-1.0) < 0.05
end

fig