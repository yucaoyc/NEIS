export load_torus_eg,
       solve_poisson_2dtorus_fft,
       MH_generate_data_from_uniform

"""
Give a 2D example on torus [0,1]^2.
"""
function load_torus_eg(N, λ, a; mode=3, prior="nonuniform")

    freq = fftfreq(N)*N
    kmat = Matrix(undef,N,N)
    for i = 1:N
        for j = 1:N
            kmat[i,j] = [freq[i], freq[j]]
        end
    end
    xgrid = Array(range(0, stop=1.0, length=N+1)[1:N])

    xxc = [xgrid[i] for i = 1:N for j = 1:N]
    yyc = [xgrid[j] for i = 1:N for j = 1:N]

    basis(x, μ) = exp(a*cos(2*π*(x[1]-μ[1]))+a*cos(2*π*(x[2]-μ[2])))

    cpts=[0.5,0.5]
    if prior == "nonuniform"
        q₀ = (x) -> basis(x, cpts)
        Z₀ = sum(map((x₁,x₂)->q₀([x₁, x₂]), xxc, yyc))*(1/N)^2
        ρ₀ = (x) -> q₀(x)/Z₀
    else
        q₀ = (x) -> 1.0
        ρ₀ = (x) -> 1.0
        Z₀ = 1.0
    end

    vpts = [cpts.+[-λ,-λ], cpts.+[λ,-λ], cpts.+[-λ,λ], cpts.+[λ,λ]]
    function q₁(x)
        if mode == 4
            return (basis(x, vpts[1]) + basis(x, vpts[2]) + basis(x, vpts[3]) + basis(x, vpts[4]))/4
        elseif mode == 3
            return (basis(x, vpts[1]) + basis(x, vpts[2]) + basis(x, vpts[3]))/3
        elseif mode == 2
            return (basis(x, vpts[1]) + basis(x, vpts[2]))/2
        else
            return basis(x, vpts[1])
        end
    end
    Z₁ = sum(map((x₁,x₂)->q₁([x₁, x₂]), xxc, yyc))*(1/N)^2
    ρ₁(x) = q₁(x)/Z₁;

    function ρdiff(x)
        return ρ₁(x) - ρ₀(x)
    end
    exact_mean = Z₁/Z₀

    U₀ = ExplicitPotential(Float64, 2, x->-log(q₀(x)))
    U₁ = ExplicitPotential(Float64, 2, x->-log(q₁(x)))

    return kmat, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, U₀, U₁, exact_mean
end


"""
Solve for V and 𝐛 =∇V for a 2D torus example on [0,1]]²  using FFT.

Suppose
V(x) = ∑ₖ f(k) exp(-2π i k ⋅ x)

Then
Δ V(x) = ∑ₖ f(k) ⟨2π i k, 2π i k⟩ exp(-2π i k⋅x) = -4π^2 ∑ₖ f(k) |k|^2 exp(-2π i k⋅x).
"""
function solve_poisson_2dtorus_fft(N::Int, xgrid::Array{T},
        ρdiff::Function, kmat::AbstractMatrix) where T<:AbstractFloat

    y = complex(zeros(N,N))
    for i = 1:N
        for j = 1:N
            y[i,j] = ρdiff([xgrid[i], xgrid[j]])
        end
    end
    coef = ifft(y)./(4*π^2*(-1)*map(norm, kmat).^2)
    coef[1,1] = 0.0;

    V_value = real(fft(coef))
    if maximum(abs.(imag(fft(coef)))) > 1.0e-4
        @warn("Possible problems in FFT")
    end

    x_ext = Array(range(0, stop=1.0, length=N+1))
    V_value_ext = vcat(hcat(V_value, V_value[:,1]), vcat(V_value[1,:], [V_value[1,1]])')
    V_interp = BilinearInterpolator(x_ext, x_ext, V_value_ext);

    function b_interp(x; h=1.0e-6)
        # we assume x is inside the domain
        dVx = (V_interp(mod(x[1]+h,1),mod(x[2],1)) - V_interp(mod(x[1]-h,1),mod(x[2],1)))/(2*h)
        dVy = (V_interp(mod(x[1],1),mod(x[2]+h,1)) - V_interp(mod(x[1],1),mod(x[2]-h,1)))/(2*h)
        return [dVx, dVy]
    end

    return coef, V_interp, b_interp, DynFix{T}(2, b_interp)
end

"""
Sampling data with distribution q₀ using Metropolis-Hasting correction methods.
We use uniform distribution on [0,1]ᵈ to generate data.
"""
function MH_generate_data_from_uniform(numsample::Int, q₀::Function, dim::Int)
    gpts = Array{Any}(undef, numsample)
    gpts[1] = rand(dim)
    new_pts_idx = Array{Any}(undef, numsample)
    new_pts_idx[1] = true

    count = 0
    for j = 2:numsample
        x_old = gpts[j-1]
        x_new = rand(dim) #uniform distribution

        accept_rate = min(1, q₀(x_new)/q₀(x_old))
        if rand() < accept_rate
            # accept
            gpts[j] = x_new
            count += 1
            new_pts_idx[j] = true
        else
            gpts[j] = x_old
            new_pts_idx[j] = false
        end
    end
    return count/numsample, gpts, new_pts_idx
end
