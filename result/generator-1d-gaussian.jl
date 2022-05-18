push!(LOAD_PATH,"../src")
using ModDyn
using ModPotential
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Printf
using ModODEIntegrate
using ModUtil
using ModGenerator
plot_setup();
folder = "assets_gen/"
~isdir(folder) ? mkdir(folder) : nothing

n = 1
ω = 1.0
σ = 0.5
U₀ = generate_Gaussian_Potential(n, [0.0], ones(n,n))
U₁ = generate_Gaussian_Potential(n, [ω], σ^2*ones(n,n))
flow = DynFix(n, x->x-[ω]/(1-σ))

T = log(σ)
ρ₀(x) = exp(-U₀.U(x))/sqrt(2*π)
ρ₁(x) = exp(-U₁.U(x))/sqrt(2*π*σ^2);

function move(xt, N, w, σ)
    dt = log(σ)/N
    for i = 1:N
        xt += flow.f(xt)*dt
    end
    return xt
end

numsample = 10^4;
N = 10^3
value = map(x->move(x,N,w,σ)[1], [randn(n) for i = 1:numsample]);

fig = histogram(value, normed=true, label="")
xmin = -2
xmax = 3
xgrid = range(xmin, stop=xmax, length=101);
plot!(xgrid, z->ρ₁([z]), color=:black, label="", linewidth=2, xlims=(xmin, xmax), size=(300,200))
savefig(fig, folder*"gaussian_empirical_rho_1.pdf")
fig
