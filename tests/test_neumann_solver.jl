using NEIS
using Test
using LinearAlgebra

N = 2^9; λ = 0.2; a = 2.0;
mode = 3; prior="uniform"
xmin = 0; xmax = 1; ymin = 0; ymax = 1;
Nx = N; Ny = N;

_, _, q₀, ρ₀, q₁, ρ₁, ρdiff, _, _, _ = load_torus_eg(N, λ, a, mode=mode, prior=prior);

ρρ₀ = (x,y)->ρ₀([x,y])
ρρ₁ = (x,y)->ρ₁([x,y]);

@time sol = solve_2d_poisson_neumann(xmin, xmax, ymin, ymax, Nx, Ny, ρρ₀, ρρ₁);

Φ = sol["Φ"]
V = sol["V"]
Divgϕ = sol["Divgϕ"]
b = sol["b"]
gradϕx = sol["gradϕx"]
gradϕy = sol["gradϕy"]
xgrid = sol["xgrid"]
ygrid = sol["ygrid"]

ϵ = 1.0e-5
@testset "poision solver" begin
    for iter = 1:100
        i = Int64(ceil(rand()*(Nx-2)))+1
        j = Int64(ceil(rand()*(Ny-2)))+1

        # test interpolator
        @test abs((Φ[i,j] - V(xgrid[i], ygrid[j]))/Φ[i,j]) < ϵ

        # test ΔV = ρdiff
        @test abs((Divgϕ[i,j] - ρdiff([xgrid[i],ygrid[j]]))/Divgϕ[i,j]) < ϵ

        # test ∇V = b.
        flow_b = b(xgrid[i], ygrid[j])
        flow_1 = (V(xgrid[i]+ϵ, ygrid[j]) - V(xgrid[i]-ϵ, ygrid[j]))/(2*ϵ)
        flow_2 = (V(xgrid[i], ygrid[j]+ϵ) - V(xgrid[i], ygrid[j]-ϵ))/(2*ϵ)
        @test norm(flow_b - [flow_1, flow_2])/norm(flow_b) < 1.0e-3
    end
    @test norm([norm(gradϕx[1,:]), norm(gradϕx[Nx,:]), norm(gradϕy[:,1]), norm(gradϕy[:,Ny])]) < ϵ
end
