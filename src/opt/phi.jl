export Φ, get_Φ

mutable struct Φ{T<:AbstractFloat}
    f::Function
    fderi::Function
    name::String
end

function get_Φ(name::String, ϵ::T) where T<:AbstractFloat
    if name == "msq" # mean-square
        return Φ{T}(x -> x.^2, x -> 2*x, name)
    elseif name == "xlogx"
        return Φ{T}(x -> (x.+ϵ).*log.(x.+ϵ), x -> log.(x .+ ϵ).+1, name)
    elseif name == "-logx"
        return Φ{T}(x -> -log.(x.+ϵ), x-> -1 ./ (x.+ϵ), name)
    else
        @error("Not implemented yet")
    end
end
