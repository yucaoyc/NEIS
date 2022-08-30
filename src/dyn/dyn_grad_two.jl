export DynNNGradTwo, init_DynNNGradTwo, init_random_DynNNGradTwo

"""
A gradient-form flow using two layer nn parameterization.    
"""
mutable struct DynNNGradTwo{T} <: DynTrain{T}
    dim::Int # dimension
    m::Int
    para_list::Array{Array{T}} # an array for (W_1, b_1, W_2, b_2 ..., W_L, b_L)
#    train_para_idx::Array{Int}
    V::Function
    f::Function # A function that take x, W_1, b_1, W_2, b_2, ⋯, W_L, b_L as inputs.
#    dyn_type::Int
    num_para::Array{Int}
    total_num_para::Int
    σ::Function
    σderi::Function
    σsec_deri::Function
    σth_deri::Function
end

function init_DynNNGradTwo(dim::Int, m::Int, W1::Array{T,2}, b1::Array{T,1}, W2::Array{T,2}, 
        σ=softplus, σderi=sigmoid, 
        σsec_deri=sigmoid_deri, 
        σth_deri=sigmoid_sec_deri) where T <: AbstractFloat

    para_list = [W1, b1, W2]
    V(x, W1, b1, W2) = W2*σ.(W1*x .+ b1)
    
    function f(x::Array{T}, W1::Matrix{T}, b1::Vector{T}, W2::Matrix{T}) 
        y = W1*x .+ b1
        yact_deri = σderi.(y)
        return @tullio F[i] := W2[j]*yact_deri[j]*W1[j,i]
    end

    function f(x::Matrix{T}, W1::Matrix{T}, b1::Vector{T}, W2::Matrix{T})
        y = W1*x .+ b1
        yact_deri = σderi.(y)
        return @tullio F[i,a] := W2[j]*yact_deri[j,a]*W1[j,i]
    end
    
    num_para = [length(θ) for θ in para_list] 
    total_num_para = sum(num_para)
    return DynNNGradTwo(dim, m, para_list, V, f, num_para, 
        total_num_para, σ, σderi, σsec_deri, σth_deri)
end

function init_random_DynNNGradTwo(dim::Int, m::Int; convert=x->Float32.(x),
        init=glorot_uniform, seed::Int=1, scale=1.0)
    Random.seed!(seed)
    W1 = convert(init(m, dim))
    b1 = convert(init(m))
    W2 = convert(scale*init(1,m))
    return init_DynNNGradTwo(dim, m, W1, b1, W2) 
end


function ∇b(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
    W1, b1, W2 = flow.para_list
    y_act = flow.σsec_deri.(W1*x.+b1)
    return @tullio grad[i,k] := W2[j]*W1[j,i]*W1[j,k]*y_act[j]
end

function divg_b(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
    W1, b1, W2 = flow.para_list
    y_act = flow.σsec_deri.(W1*x.+b1)
    return @tullio grad := W2[j]*W1[j,i]*W1[j,i]*y_act[j]     
end

function grad_divg_b(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
    W1, b1, W2 = flow.para_list
    W1sq = @tullio W1sq[j] := (W1[j,i])^2
    y_act = flow.σth_deri.(W1*x .+ b1)
    grad = @tullio grad[l] := W2[j]*W1sq[j]*y_act[j]*W1[j,l]
    return grad
end

function grad_b_wrt_para_part(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
    n = flow.dim
    m = flow.m
    W1, b1, W2 = flow.para_list
    y = W1*x.+b1
    y_act_1 = flow.σderi.(y)
    y_act_2 = flow.σsec_deri.(y)
    Id = Matrix{T}(1.0I,n,n)
    
    FW2 = @tullio FW2[i,j] := y_act_1[j]*W1[j,i]
    Fb1 = @tullio Fb1[i,k] := W2[k]*W1[k,i]*y_act_2[k]
    FW1 = @tullio FW1[i,l,μ] := W2[l]*y_act_2[l]*W1[l,i]*x[μ] + W2[l]*y_act_1[l]*Id[i,μ]
    FW1 = reshape(FW1, (n, n*m))
    
    return FW1, Fb1, FW2
end

function grad_b_wrt_para(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
   return hcat(grad_b_wrt_para_part(flow, x)...) 
end

function grad_divg_wrt_para_part(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
    n = flow.dim
    W1, b1, W2 = flow.para_list
    y = W1*x.+b1  
    W1sq = @tullio W1sq[j] := (W1[j,i])^2
    y_act_2 = flow.σsec_deri.(y)
    y_act_3 = flow.σth_deri.(y)
    
    FW2 = @tullio FW2[j] := W1sq[j]*y_act_2[j]
    Fb1 = @tullio Fb1[k] := W2[k]*W1sq[k]*y_act_3[k]
    FW1 = @tullio FW1[l,μ] := 2*W2[l]*y_act_2[l]*W1[l,μ] + W2[l]*W1sq[l]*y_act_3[l]*x[μ]
    FW1 = vec(FW1)
    
    return FW1, Fb1, FW2
end

function grad_divg_wrt_para(flow::DynNNGradTwo{T}, x::Array{T}) where T<:AbstractFloat
   return vcat(grad_divg_wrt_para_part(flow, x)...) 
end

# matrix version.

function ∇b(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    W1, b1, W2 = flow.para_list
    y_act = flow.σsec_deri.(W1*x.+b1)
    return @tullio grad[i,k,a] := W2[j]*W1[j,i]*W1[j,k]*y_act[j,a]
end

function divg_b(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    W1, b1, W2 = flow.para_list
    y_act = flow.σsec_deri.(W1*x.+b1)
    return @tullio grad[a] := W2[j]*W1[j,i]*W1[j,i]*y_act[j, a]     
end

function grad_divg_b(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    W1, b1, W2 = flow.para_list
    W1sq = @tullio W1sq[j] := (W1[j,i])^2
    y_act = flow.σth_deri.(W1*x .+ b1)
    grad = @tullio grad[l,a] := W2[j]*W1sq[j]*y_act[j,a]*W1[j,l]
    return grad
end

function grad_b_wrt_para_part(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    num_particle = size(x,2)
    n = flow.dim
    m = flow.m
    W1, b1, W2 = flow.para_list
    y = W1*x.+b1
    y_act_1 = flow.σderi.(y)
    y_act_2 = flow.σsec_deri.(y)
    Id = Matrix{T}(1.0I,n,n)
    
    FW2 = @tullio FW2[i,j,a] := y_act_1[j,a]*W1[j,i]
    Fb1 = @tullio Fb1[i,k,a] := W2[k]*W1[k,i]*y_act_2[k,a]
    FW1 = @tullio FW1[i,l,μ, a] := W2[l]*y_act_2[l,a]*W1[l,i]*x[μ,a] + W2[l]*y_act_1[l,a]*Id[i,μ]
    FW1 = reshape(FW1, (n, n*m, num_particle))
    
    return FW1, Fb1, FW2
end

function grad_b_wrt_para(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    n = flow.dim
    m = flow.m
    num_particle = size(x,2)
    FW1, Fb1, FW2 = grad_b_wrt_para_part(flow, x)
    A = zeros(T, n, flow.total_num_para, num_particle)
    A[:,1:(m*n),:] = FW1
    A[:,(m*n+1):(m*(n+1)),:] = Fb1
    A[:,(m*(n+1)+1):end,:] = FW2
    return A
end

function grad_divg_wrt_para_part(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    n = flow.dim
    m = flow.m
    num_particle = size(x,2)
    
    W1, b1, W2 = flow.para_list
    y = W1*x.+b1  
    W1sq = @tullio W1sq[j] := (W1[j,i])^2
    y_act_2 = flow.σsec_deri.(y)
    y_act_3 = flow.σth_deri.(y)
    
    FW2 = @tullio FW2[j,a] := W1sq[j]*y_act_2[j,a]
    Fb1 = @tullio Fb1[k,a] := W2[k]*W1sq[k]*y_act_3[k,a]
    FW1 = @tullio FW1[l,μ,a] := 2*W2[l]*y_act_2[l,a]*W1[l,μ] + W2[l]*W1sq[l]*y_act_3[l,a]*x[μ,a]
    FW1 = reshape(FW1, (n*m, num_particle))
    
    return FW1, Fb1, FW2
end

function grad_divg_wrt_para(flow::DynNNGradTwo{T}, x::Matrix{T}) where T<:AbstractFloat
    n = flow.dim
    m = flow.m
    num_particle = size(x,2)
    
    FW1, Fb1, FW2 = grad_divg_wrt_para_part(flow, x)
    A = zeros(T, flow.total_num_para, num_particle)
    A[1:(m*n),:] = FW1
    A[(m*n+1):(m*(n+1)),:] = Fb1
    A[(m*(n+1)+1):end,:] = FW2
    return A
end
