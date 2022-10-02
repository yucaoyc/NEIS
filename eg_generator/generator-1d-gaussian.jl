##############################
# The example in Appendix G.2.1 of arXiv:2206.09908v1
##############################
using NEIS
using LinearAlgebra
using Plots, Plots.PlotMeasures, LaTeXStrings, Printf
gr()
plot_setup()
folder = "assets/"
~isdir(folder) ? mkdir(folder) : nothing
figsize = (350,300)

n = 1
ω = 1.0
σ = 0.5
U₀ = Gaussian(n, [0.0], ones(n,n))
U₁ = Gaussian(n, [ω], σ^2*ones(n,n))
flow = DynFix{Float64}(n, x->x-[ω]/(1-σ))

τ = log(σ)
ρ₀(x) = exp(-U₀(x))/sqrt(2*π)
ρ₁(x) = exp(-U₁(x))/sqrt(2*π*σ^2);

# move according to the flow above for time τ.
function move(xt, N, w, σ)
    dt = τ/N
    for i = 1:N
        xt += flow(xt)*dt
    end
    return xt
end

numsample = 10^4;
N = 10^3
value = map(x->move(x,N,w,σ)[1], [randn(n) for i = 1:numsample]);

fig = histogram(value, normed=true, label="")
xmin = -2; xmax = 3
xgrid = range(xmin, stop=xmax, length=101);
plot!(xgrid, z->ρ₁([z]), color=:black, label="", linewidth=2, xlims=(xmin, xmax), size=figsize, margin=20px)
savefig(fig, folder*"gaussian_empirical_rho_1.pdf")

fig
