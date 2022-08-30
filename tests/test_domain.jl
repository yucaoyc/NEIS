using NEIS
using LinearAlgebra
using PyPlot

function get_domain(case)
    if case == 1
        Ω₁ = domain_ball(1.0)
        Ω₂ = domain_ball(1.0)
        Ω = prod_domain(Ω₁, Ω₂, 1, 1)
    else
        lb = [-1, -0.5]; ub = [1, 0.5]
        Ω = x->domain_rectangle(x, lb, ub)
    end
    return Ω
end

for case = 1:2
    Ω = get_domain(case)

    N = 100
    x = Array(range(-2, stop=2.0, length=N))
    y = x
    XX, YY = meshgrid(x,y)

    XX = zeros(N, N)
    YY = zeros(N, N)
    ZZ = zeros(N, N)

    for i = 1:N
        for j = 1:N
            XX[i,j] = x[i]
            YY[i,j] = y[j]
            ZZ[i,j] = Int64(Ω([x[i], y[j]]))
        end
    end
    PyPlot.figure(figsize=(6,3))
    PyPlot.surf(XX, YY, ZZ)
end
