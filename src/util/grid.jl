export rescaleidx, shortest_distance_2d_grid,
    get_grids, get_sparse_grids, meshgrid

"""
Return the meshgrid
"""
function meshgrid(xgrid::Array{T}, ygrid::Array{T}) where T <: AbstractFloat
    Nx = length(xgrid)
    Ny = length(ygrid)
    CXX = zeros(Nx, Ny)
    CYY = zeros(Nx, Ny)
    for i = 1:Nx
        for j = 1:Ny
            CXX[i,j] = xgrid[i]
            CYY[i,j] = ygrid[j]
        end
    end
    return CXX, CYY
end

"""
Generate grid points for training.
"""
function get_grids(Δx::T, Δy::T, xmin::T, xmax::T, ymin::T, ymax::T) where T <: AbstractFloat
    ΔA = (Δx*Δy)
    xx = Vector(xmin:Δx:xmax)
    yy = Vector(ymin:Δy:ymax)
    grid_pts = []
    for i = 1:length(xx)
        for j = 1:length(yy)
            push!(grid_pts, [xx[i],yy[j]])
        end
    end
    return ΔA, xx, yy, grid_pts
end

"""
Get grid points with large separation distance.
By default, only (12+1) points are used in each dimension.
"""
function get_sparse_grids(xmin::T, xmax::T, ymin::T, ymax::T; sp::Int=12) where T <: AbstractFloat
    # sparse points for plotting vector fields.
    sparse_xx = Vector(xmin:(xmax-xmin)/sp:xmax)
    sparse_yy = Vector(ymin:(ymax-ymin)/sp:ymax)
    xxb = [x for x in sparse_xx for y in sparse_yy]
    yyb = [y for x in sparse_xx for y in sparse_yy]
    return xxb, yyb
end

"""
Transform the index a 2D grid point (i,j).
"""
function rescaleidx(ij::Tuple, M::Int)
    i, j = ij
    return (j-1)*M + i
end

"""
Inverse function of the last function.
""" 
function rescaleidx(idx::Int, M::Int) 
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
function shortest_distance_2d_grid(ij::Tuple, kl::Tuple, M::Int)
    i, j = ij
    k, l = kl
    
    dik = abs(i-k)
    djl = abs(j-l)
    return sqrt(min(dik, M - dik)^2 + min(djl, M - djl)^2)
end
