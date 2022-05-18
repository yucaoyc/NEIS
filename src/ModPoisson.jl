module ModPoisson

using Documenter
using FFTW
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using BasicInterpolators
using ModDyn:DynFix

export load_torus_eg,
       lowergamma, 
       exact_dynb_poisson,
       solve_poisson_2dtorus_fft,
       MH_generate_data_from_uniform, 
       solve_poisson, 
       get_grad_phi

function lowergamma(a,z)
   gamma(a) - gamma(a,z) 
end

function exact_dynb_poisson_one_mode(n, w, μ, σsq)
    b(x) = π^(-n/2)*2^(-1)*(
        sum([w[i]*norm(x .- μ[i])^(-n)*lowergamma(n/2, norm(x .- μ[i])^2/(2*σsq[i]))*(x .- μ[i]) 
                for i = 1:length(w)]))
end
    
function exact_dynb_poisson(n, w1, v1, σ1sq, w2, v2, σ2sq)
    b1 = exact_dynb_poisson_one_mode(n, w1, v1, σ1sq)
    b2 = exact_dynb_poisson_one_mode(n, w2, v2, σ2sq)
    #return x-> b1(x) .- b2(x)
    return x-> b2(x) .- b1(x)
end

"""
    Give a 2D example on torus.
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
    #println(sum(map((x₁,x₂) -> ρ₀([x₁, x₂]), xxc, yyc))*(1/N)^2)

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
    #println(sum(map((x₁,x₂) -> ρ₁([x₁, x₂]), xxc, yyc))*(1/N)^2)

    ρdiff(x) = ρ₁(x) - ρ₀(x)
    exact_mean = Z₁/Z₀
    
    return kmat, xgrid, q₀, ρ₀, q₁, ρ₁, ρdiff, exact_mean
end


"""
Solve for ``V`` and ``\\boldsymbol{b}=\\nabla V`` for a 2D torus example on ``[0,1]^2`` using FFT.

Suppose
``V(\\vec{x}) = \\sum_{\\vec{k}} f(\\vec{k}) e^{-2\\pi i \\vec{k}\\cdot \\vec{x}}.``

Then
```math
\\Delta V(\\vec{x}) = \\sum_{\\vec{k}} f(\\vec{k}) \\langle 2\\pi i \\vec{k}, 2\\pi i \\vec{k}\\rangle
e^{-2\\pi i \\vec{k}\\cdot \\vec{x}} =
-4 \\pi^2 \\sum_{\\vec{k}} f(\\vec{k}) \\lvert \\vec{k}\\rvert^2 e^{-2\\pi i \\vec{k}\\cdot \\vec{x}}
```
"""
function solve_poisson_2dtorus_fft(N, xgrid, ρdiff, kmat)
    
    y = complex(zeros(N,N))
    for i = 1:N
        for j = 1:N
            y[i,j] = ρdiff([xgrid[i], xgrid[j]])
        end
    end
    coef = ifft(y)./(4*π^2*(-1)*map(norm, kmat).^2)
    coef[1,1] = 0.0;

    V_value = real(fft(coef))
    x_ext = Array(range(0, stop=1.0, length=N+1))
    V_value_ext = vcat(hcat(V_value, V_value[:,1]), vcat(V_value[1,:], [V_value[1,1]])')
    V_interp = BilinearInterpolator(x_ext, x_ext, V_value_ext);

    function b_interp(x; h=1.0e-6)
        # we assume x is inside the domain
        dVx = (V_interp(mod(x[1]+h,1),mod(x[2],1)) - V_interp(mod(x[1]-h,1),mod(x[2],1)))/(2*h)
        dVy = (V_interp(mod(x[1],1),mod(x[2]+h,1)) - V_interp(mod(x[1],1),mod(x[2]-h,1)))/(2*h)
        return [dVx, dVy]
    end
    
    return coef, V_interp, b_interp, DynFix(2, b_interp)
end

"""
    Sampling data with distribution q₀ using Metropolis-Hasting correction methods by uniform distribution on [0,1]^d
"""
function MH_generate_data_from_uniform(numsample, q₀, dim)
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

"""
    Test if the index i is the boundary point where Nx is the number of grid points.
"""
function boundary(Nx, i)
   if i == 0
        newi = 2
    elseif i == Nx+1
        newi = Nx-1
    else
        newi = i
    end
    return newi
end

"""
    Solve the Poisson's equation Δϕ = \rho_1 - \rho_0 with neumann boundary condition on a 2D rectangular region.

    ϕ contains 2D function values.
"""
function solve_poisson(xmin, xmax, ymin, ymax, Nx, Ny, ρ₀, ρ₁)
    
    Δx = (xmax-xmin)/(Nx-1)
    Δy = (ymax-ymin)/(Ny-1)

    num_elem = Nx*Ny

    Ci = zeros(num_elem*5)
    Cj = zeros(num_elem*5)
    Cv = zeros(num_elem*5)
    Value = zeros(num_elem)
    count = 0

    for i = 1:(Nx)
        for j = 1:(Ny)
            idx = (i-1)*Ny + j
            # left point
            count += 1
            Ci[count] = idx
            if i > 1
                Cj[count] = idx - Ny
            else
                Cj[count] = idx + Ny
            end
            Cv[count] = 1/(2*Δx^2)
            # right point
            count += 1
            Ci[count] = idx
            if i < Nx
                Cj[count] = idx + Ny
            else
                Cj[count] = idx - Ny
            end
            Cv[count] = 1/(2*Δx^2)
            # upper point
            count += 1
            Ci[count] = idx
            if j < Ny
                Cj[count] = idx + 1
            else
                Cj[count] = idx - 1
            end
            Cv[count] = 1/(2*Δy^2)
            # lower point
            count += 1
            Ci[count] = idx
            if j > 1
                Cj[count] = idx - 1
            else
                Cj[count] = idx + 1
            end
            Cv[count] = 1/(2*Δy^2)      
            # it self
            count += 1
            Ci[count] = idx
            Cj[count] = idx
            Cv[count] = -1/(Δx^2) - 1/(Δy^2)

            xi = (i-1)*Δx + xmin
            yj = (j-1)*Δy + ymin

            Value[idx] = ρ₁(xi, yj) - ρ₀(xi, yj)
        end
    end

    C = sparse(Ci, Cj, Cv);

    M = vcat(hcat(C, ones(num_elem,1)), hcat(ones(1, num_elem), zeros(1,1)))
    V = vcat(Value, zeros(1));
    ϕ = M\V;

    x = range(xmin, stop=xmax, length=Nx)
    y = range(ymin, stop=ymax, length=Ny)
    Φ = zeros(Nx, Ny)
    CXX = zeros(Nx, Ny)
    CYY = zeros(Nx, Ny)

    for i = 1:Nx
        for j = 1:Ny
            idx = (i-1)*Ny + j
            Φ[i,j] = ϕ[idx]
            CXX[i,j] = (i-1)*Δx + xmin
            CYY[i,j] = (j-1)*Δy + ymin
        end
    end

    return Δx, Δy, x, y, CXX, CYY, Φ
    
end

"""
    Return ∇_xϕ, ∇_yϕ, Δϕ on the domain [0,1]^2
    Nx, Ny are numbers of grid points
    Δx, Δy are grid sizes.
"""
function get_grad_phi(Nx, Ny, Φ, Δx, Δy)
    
    gradϕx = zeros(Nx, Ny)
    gradϕy = zeros(Nx, Ny)
    Divgϕ = zeros(Nx, Ny)

    for i = 1:Nx
        for j = 1:Ny
            gradϕx[i,j] = (Φ[boundary(Nx,i+1), boundary(Ny, j)] - Φ[boundary(Nx,i-1), boundary(Ny, j)])/(2*Δx)
            gradϕy[i,j] = (Φ[boundary(Nx,i), boundary(Ny, j+1)] - Φ[boundary(Nx,i), boundary(Ny, j-1)])/(2*Δy)
            Divgϕ[i,j] = ((Φ[boundary(Nx,i+1), boundary(Ny, j)] + Φ[boundary(Nx,i-1), boundary(Ny, j)] - 2*Φ[i,j])/(2*Δx^2) 
                          + (Φ[boundary(Nx,i), boundary(Ny, j+1)] + Φ[boundary(Nx,i), boundary(Ny, j-1)] - 2*Φ[i,j])/(2*Δy^2))
        end
    end
    return gradϕx, gradϕy, Divgϕ
end

end
