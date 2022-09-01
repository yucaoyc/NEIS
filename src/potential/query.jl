export QueryNumber, get_query_stat, print_query_stat, reset_query_stat, verify_budget

# A data type for query
QueryNumber = Union{UInt128,Threads.Atomic{UInt128}}

"""
set-up a QueryNumber based on safe mode.
"""
function set_query_number(a::Int; safe::Bool=false)
    a < 0 ? error("a must be non-negative") : nothing
    if safe
        return Threads.Atomic{UInt128}(a)
    else
        return UInt128(a)
    end
end

"""
An auxillary function used to update the query number in safe mode.
"""
function query_add!(x::Threads.Atomic{UInt128}, y::Int)
    Threads.atomic_add!(x, UInt128(y))
end

"""
Given counting mode, return safe mode.
"""
function get_safe_mode(count_mode::Symbol)
    if (count_mode == :unsafe_count) || (count_mode == :not_count)
        safe = false
    elseif count_mode == :safe_count
        safe = true
    else
        @error("count_mode is incorrect!")
    end
    return safe
end

########################################
# update query statistics
#

function reset_query_stat(p::Potential)
    if typeof(p.query_u) <: Threads.Atomic
        p.query_u[] = UInt128(0)
        p.query_gradu[] = UInt128(0)
        p.query_hessu[] = UInt128(0)
        p.query_laplaceu[] = UInt128(0)
    else
        p.query_u = UInt128(0)
        p.query_gradu = UInt128(0)
        p.query_hessu = UInt128(0)
        p.query_laplaceu = UInt128(0)
    end

    return nothing
end

function query_u_add!(p::Potential, v::Int)
    if p.count_mode == :unsafe_count
        p.query_u += v
    elseif p.count_mode == :safe_count
        query_add!(p.query_u, v)
    elseif p.count_mode == :not_count
        nothing
    else
        @error("incorrect mode!")
    end
end

function query_gradu_add!(p::Potential, v::Int)
    if p.count_mode == :unsafe_count
        p.query_gradu += v
    elseif p.count_mode == :safe_count
        query_add!(p.query_gradu, v)
    elseif p.count_mode == :not_count
        nothing
    else
        @error("incorrect mode!")
    end
end

function query_hessu_add!(p::Potential, v::Int)
    if p.count_mode == :unsafe_count
        p.query_hessu += v
    elseif p.count_mode == :safe_count
        query_add!(p.query_hessu, v)
    elseif p.count_mode == :not_count
        nothing
    else
        @error("incorrect mode!")
    end
end

function query_laplaceu_add!(p::Potential, v::Int)
    if p.count_mode == :unsafe_count
        p.query_laplaceu += v
    elseif p.count_mode == :safe_count
        query_add!(p.query_laplaceu, v)
    elseif p.count_mode == :not_count
        nothing
    else
        @error("incorrect mode!")
    end
end

function U(p::Potential{T}, x::Matrix{T}) where T<:AbstractFloat
    query_u_add!(p, size(x,2))
    return _U(p, x)
end

function U(p::Potential{T}, x::Vector{T}) where T<:AbstractFloat
    query_u_add!(p, 1)
    return _U(p, x)
end

function ∇U(p::Potential{T}, x::Matrix{T}) where T<:AbstractFloat
    query_gradu_add!(p, size(x,2))
    return _∇U(p, x)
end

function ∇U(p::Potential{T}, x::Vector{T}) where T<:AbstractFloat
    query_gradu_add!(p, 1)
    return _∇U(p, x)
end

function HessU(p::Potential{T}, x::Matrix{T}) where T<:AbstractFloat
    query_hessu_add!(p, size(x,2))
    return _HessU(p, x)
end

function HessU(p::Potential{T}, x::Vector{T}) where T<:AbstractFloat
    query_hessu_add!(p, 1)
    return _HessU(p, x)
end

function LaplaceU(p::Potential{T}, x::Matrix{T}) where T<:AbstractFloat
    query_laplaceu_add!(p, size(x,2))
    return _LaplaceU(p, x)
end

function LaplaceU(p::Potential{T}, x::Vector{T}) where T<:AbstractFloat
    query_laplaceu_add!(p, 1)
    return _LaplaceU(p, x)
end

########################################
# Statistics
#

function get_query_stat(p::Potential)
    if typeof(p.query_u) <: Number
        return [p.query_u, p.query_gradu, p.query_hessu, p.query_laplaceu]
    elseif typeof(p.query_u) <: Threads.Atomic
        return [p.query_u[], p.query_gradu[], p.query_hessu[], p.query_laplaceu[]]
    else
        @error("query data type does not match!")
    end
end

"""
If type == :nonzero, we only print non-zero entries;
If type == :full, we print all entries.
"""
function print_query_stat(p::Potential; type=:nonzero)
    a, b, c, d = get_query_stat(p)
    if type == :full
        @printf("query (U): %10s\n", datasize(a))
        @printf("query (∇U): %9s\n", datasize(b))
        @printf("query (∇²U): %8s\n", datasize(c))
        @printf("query (ΔU): %9s\n", datasize(d))
    elseif type == :nonzero
        a > 0 ? @printf("query (U): %10s\n", datasize(a)) : nothing
        b > 0 ? @printf("query (∇U): %9s\n", datasize(b)) : nothing
        c > 0 ? @printf("query (∇²U): %8s\n", datasize(c)) : nothing
        d > 0 ? @printf("query (ΔU): %9s\n", datasize(d)) : nothing
    else
        @error("Please use either :nonzero or :full in print_query_stat!")
    end
end

########################################
# Verify query
#
function verify_budget(U::Potential, query_budget::Int; lb=0.98, ub=1.01)
    empirical = maximum(get_query_stat(U))
    if empirical < query_budget*lb
        @warn("Use fewer queries than allowed!")
    elseif empirical > query_budget*ub
        @error("Use too many queries than allowed!")
    else
        # everything is good
        printstyled("Pass query test!\n", color=:green)
    end
end
