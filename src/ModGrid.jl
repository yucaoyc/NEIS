module ModGrid

using LinearAlgebra

export rescaleidx, shortest_distance_2d_grid

"""
    Re-index a 2D grid point (i,j)
"""
function rescaleidx(ij::Tuple, M::Int64)
    i, j = ij
    return (j-1)*M + i
end

"""
    Inverse function of the last function.
""" 
function rescaleidx(idx::Int64, M::Int64) 
    if idx <= 0 || idx > M^2
        error("wrong index")
    else
        j = Int64(ceil(idx/M))
        i = idx - (j-1)*M
        return i, j
    end
end

"""
    Shortest distance for 2D indices ij and kl on a torus.
    The torus is integer grid points on [0, M]^2.
"""
function shortest_distance_2d_grid(ij, kl, M)
    i, j = ij
    k, l = kl
    
    dik = abs(i-k)
    djl = abs(j-l)
    return sqrt(min(dik, M - dik)^2 + min(djl, M - djl)^2)
end


end
