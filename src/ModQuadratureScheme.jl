module ModQuadratureScheme

export QuadratureScheme1D, Gauss_Legendre, Trapezoidal

    mutable struct QuadratureScheme1D
        # This data structure is used to store quadrature schemes on [-1,1]
        # if end points are not used, we still add them with weight 0.
        ξ::Array{Float64,1}
        w::Array{Float64,1}

        # an inner constructor
        function QuadratureScheme1D(ξ::Array{Float64,1}, w::Array{Float64,1})
            ϵ = 1.0e-14
            n = length(ξ) 

            # error handling
            if n <= 0
                error("QuadratureScheme1D: vector must be non-empty.")
            end

            if n != length(w)
                error("QuadratureScheme1D: vector lengths must match.")
            end

            if abs(sum(w) - 2.0) > ϵ
                error("QuadratureScheme1D: weight must sum up to 2.")
            end

            if minimum(w) <= -ϵ
                error("QuadratureScheme1D: weight must be non-negative.")
            end

            sort!(ξ)
            if ξ[1] <= -1.0 - ϵ
                error("QuadratureScheme1D: sample points must be above -1.")
            end

            if ξ[n] >= 1.0 + ϵ
                error("QuadratureScheme1D: sample points must be below 1.")
            end

            # add end points if necessary
            if abs(ξ[1] - (-1.0)) > ϵ
                pushfirst!(ξ, -1.0)
                pushfirst!(w, 0.0)
            end

            if abs(ξ[length(ξ)] - 1.0) > ϵ
                push!(ξ, 1.0)
                push!(w, 0.0)
            end

            return new(ξ, w)
        end # end of the constructor
    end


    function Gauss_Legendre(n)
        # Ref: https://en.wikipedia.org/wiki/Gaussian_quadrature

        if n == 1
            return QuadratureScheme1D([0.0],[2.0])
        elseif n == 2
            return QuadratureScheme1D([-1/sqrt(3), 1/sqrt(3)],[1.0,1.0])
        elseif n == 3
            return QuadratureScheme1D([-sqrt(3/5), 0.0, sqrt(3/5)], [5/9, 8/9, 5/9])
        elseif n == 4
            ξ = [-sqrt(3/7+2/7*sqrt(6/5)), -sqrt(3/7-2/7*sqrt(6/5)), sqrt(3/7-2/7*sqrt(6/5)), sqrt(3/7+2/7*sqrt(6/5))]
            w = [1/2-sqrt(30)/36, 1/2+sqrt(30)/36, 1/2+sqrt(30)/36, 1/2-sqrt(30)/36]
            return QuadratureScheme1D(ξ,w)
        else
            @warn "n >= 5 is not supported yet!"
            return
        end

    end

    
    function Trapezoidal(mat::Matrix{Float64}, h::Float64)
        # This function implements the trapezoidal rule for a matrix-valued data
        # we sum over the dimension 2.
        mat_dim = size(mat,2)
        return h*(sum(mat,dims=2) - 0.5*(mat[:,1]+mat[:,mat_dim])) 
    end

    function Trapezoidal(mat::Array{Float64,1}, h::Float64)
        # This function implements the trapezoidal rule for data stored in a vector
        mat_dim = length(mat)
        return h*(sum(mat) - 0.5*(mat[1]+mat[mat_dim])) 
    end


end