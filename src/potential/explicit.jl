export ExplicitPotential

mutable struct ExplicitPotential{T<:AbstractFloat} <: Potential{T}
    dim::Int
    u::Function
    query_u::QueryNumber
    query_gradu::QueryNumber
    query_hessu::QueryNumber
    query_laplaceu::QueryNumber
    count_mode::Symbol
end

function ExplicitPotential(T::DataType, dim::Int, u::Function; count_mode=:safe_count)
    safe = get_safe_mode(count_mode)
    return ExplicitPotential{T}(dim, u,
                    set_query_number(0, safe=safe),
                    set_query_number(0, safe=safe),
                    set_query_number(0, safe=safe),
                    set_query_number(0, safe=safe),
                    count_mode)
end

function _U(p::ExplicitPotential, x::Array{T}) where T<: AbstractFloat
    return p.u(x)
end
