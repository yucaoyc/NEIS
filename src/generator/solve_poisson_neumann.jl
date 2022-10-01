export solve_2d_poisson_neumann, get_grad_phi

function boundary(Nx::Int, i::Int)
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
Solve the Poisson's equation
Δ ϕ = ρ₁ - ρ₀
with neumann boundary condition on a 2D rectangular region [xmin, xmax]×[ymin, ymax].

We return the solution with ∫ ϕ dx = 0 as there is a degree of freedom to choose ϕ up to some constant.

ϕ contains 2D function values.
This ϕ is not the energy landscape ϕ. This is a different variable!
"""
function solve_2d_poisson_neumann(xmin::Number, xmax::Number, ymin::Number, ymax::Number, Nx::Int, Ny::Int,
        ρ₀::Function, ρ₁::Function)

    # grid points are x₁=xmin, x₂, ⋯ , x_{Nₓ} = xmax.
    #  similarly for the y-axis.
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
    if abs(ϕ[end]) > 1.0e-10
        @warn("Possible problem in solving Poisson equaiton (Neumann condition)!")
    end

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

    gradϕx, gradϕy, Divgϕ = get_grad_phi(Nx, Ny, Φ, Δx, Δy)
    # use interpolators
    V_interp = BilinearInterpolator(x, y, Φ)
    bx = BilinearInterpolator(x, y, gradϕx)
    by = BilinearInterpolator(x, y, gradϕy)

    function b_interp(x)
        return [bx(x[1],x[2]), by(x[1],x[2])]
    end

    function b_interp(x,y)
        return b_interp([x,y])
    end

    return Dict{String, Any}("Δx" => Δx, "Δy" => Δy, "xgrid"=> x, "ygrid"=>y,
                              "CXX"=>CXX, "CYY"=>CYY, "Φ"=>Φ,
                              "gradϕx"=>gradϕx, "gradϕy"=>gradϕy,
                              "Divgϕ"=>Divgϕ,
                              "V" => V_interp,
                              "bx" => bx, "by" => by, "b" => b_interp)
end

"""
Return ∇_x ϕ, ∇_y ϕ, Δϕ where ϕ is solved above.
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
            Divgϕ[i,j] = ((Φ[boundary(Nx,i+1), boundary(Ny, j)] +
                           Φ[boundary(Nx,i-1), boundary(Ny, j)] - 2*Φ[i,j])/(2*Δx^2)
                          + (Φ[boundary(Nx,i), boundary(Ny, j+1)] +
                             Φ[boundary(Nx,i), boundary(Ny, j-1)] - 2*Φ[i,j])/(2*Δy^2))
        end
    end

    return gradϕx, gradϕy, Divgϕ
end
