using NEIS
using Plots
using Plots.PlotMeasures

dim = 2
N = 2^9
λ = 0.2
a = 2.0
mode=3
kmat, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, _, _, _ = load_torus_eg(N, λ, a, mode=mode);
numsample = 10^6
percent, gpts, _ = MH_generate_data_from_uniform(numsample, q₁, dim)

coordinate_x = [item[1] for item in gpts]
coordinate_y = [item[2] for item in gpts]
f1 = contour(xgrid, xgrid, (x,y)->ρ₁([x,y]), fill=true, size=(400,300))
f2 = histogram2d(coordinate_x, coordinate_y, normed=true, nbinsx=30, nbinsy=30, size=(400,300))
plot(f1, f2, size=(800,300), margin=20px)
