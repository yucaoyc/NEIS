export Trapezoidal

"""
This function implements the trapezoidal rule for a matrix-valued data.
We sum over the dimension 2.
"""
function Trapezoidal(mat::Matrix{T}, h::T) where T <: AbstractFloat
    mat_dim = size(mat,2)
    return h*(sum(mat,dims=2) - T(0.5)*(mat[:,1]+mat[:,mat_dim])) 
end

"""
This function implements the trapezoidal rule for data stored in a vector.
"""
function Trapezoidal(mat::Array{T,1}, h::T) where T <: AbstractFloat
    mat_dim = length(mat)
    return h*(sum(mat) - T(0.5)*(mat[1]+mat[mat_dim])) 
end
