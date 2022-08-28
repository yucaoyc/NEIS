export ∇b, 
       divg_b, 
       grad_divg_b,
       grad_b_wrt_para,
       grad_divg_wrt_para,
       grad_b_wrt_para_part,
       grad_divg_wrt_para_part

export DynNNGenericTwo, 
    init_DynNNGenericTwo,
    init_random_DynNNGenericTwo

"""
A Generic Two layer nn parameterization of flow 𝐛.
"""
mutable struct DynNNGenericTwo{T} <: DynTrain{T}
    dim::Int64 # dimension
    m::Int64 # layer width
    para_list::Array{Array{T}} # (W_1, b_1, W_2, b2)
#    train_para_idx::Array{Int} # [1, 2, 3, 4] all parameters can be trained.
    f::Function # A function that take x, W_1, b_1, W_2, b_2, ⋯, W_L, b_L as inputs.
    num_para::Array{Int} # a list that contains the number of parameters for each item in para_list.
    total_num_para::Int
    σ::Function 
    σderi::Function
    σsec_deri::Function
end

function init_DynNNGenericTwo(dim::Int, m::Int, 
        W1::Array{T}, b1::Array{T}, W2::Array{T}, b2::Array{T}; 
        σ = softplus, σderi = sigmoid, σsec_deri = sigmoid_deri) where T <: AbstractFloat
    f(x, W1, b1, W2, b2) = W2*σ.(W1*x .+ b1).+b2
    para_list = [W1, b1, W2, b2]
    #num_para = [dim*m, m, dim*m, dim]
    num_para = [length(θ) for θ in para_list] 
    total_num_para = sum(num_para)
    #return DynNNGenericTwo(dim, m, para_list , [1,2,3,4], f,
    #    num_para, total_num_para, σ, σderi, σsec_deri)
    return DynNNGenericTwo(dim, m, para_list, f,
        num_para, total_num_para, σ, σderi, σsec_deri)
end

"""
Initialize a random DynNNGenericTwo.
"""
function init_random_DynNNGenericTwo(dim::Int, m::Int; 
        convert=x->Float32.(x), init=glorot_uniform, seed::Int=1,
        σ = softplus, σderi = sigmoid, σsec_deri = sigmoid_deri)
    n = dim
    Random.seed!(seed)
    W1 = convert(init(m,n))
    b1 = convert(init(m))
    W2 = convert(init(n,m))
    b2 = convert(init(n))
    return init_DynNNGenericTwo(dim, m, W1, b1, W2, b2, 
                                σ=σ, σderi=σderi,σsec_deri=σsec_deri)
end

"""
Compute ∂_θ 𝐛_θ (x).
"""
function grad_b_wrt_para_part(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    num_particle = size(x,2)
    W1, b1, W2, b2 = flow.para_list
    n = flow.dim
    m = flow.m

    y1 = W1*x .+ b1 
    y1act = flow.σ.(y1);

    Fb1 = @tullio Fb1[i,k,a] := W2[i,k]*flow.σderi.(y1)[k,a]
    Fw1 = @tullio Fw1[a,b,c,d] := Fb1[a,b,d] * x[c,d]
    Fw1 = reshape(Fw1, (n, n*m ,num_particle))

    Id = Matrix{T}(1.0I,n,n)
    Fb2 = Id
    Fw2 = @tullio Fw2[i,k,l,a] := Id[i,k]*y1act[l,a]
    Fw2 = reshape(Fw2, (n, n*m, num_particle))
    
    return Fw1, Fb1, Fw2, Fb2
end

function update_grad_b_wrt_para!(A::Array{T}, 
        flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    n = flow.dim
    m = flow.m
    # compute ∇_θ b(x) and update it to A.
    Fw1, Fb1, Fw2, Fb2 = grad_b_wrt_para_part(flow, x)
    l1 = n*m
    l2 = l1 + m
    l3 = l2 + n*m
    l4 = l3 + n
    A[:,1:l1,:] .+= Fw1
    A[:,(l1+1):l2,:] .+= Fb1
    A[:,(l2+1):l3,:] .+= Fw2
    A[:,(l3+1):l4,:] .+= Fb2
end

function grad_b_wrt_para(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    num_particle = size(x,2)
    A = zeros(T, flow.dim, flow.total_num_para, num_particle)
    update_grad_b_wrt_para!(A, flow, x)
    return A
end

function grad_b_wrt_para(flow::DynNNGenericTwo{T}, x::Vector{T}) where T <: AbstractFloat
   return grad_b_wrt_para(flow, reshape(x,(length(x),1)))[:,:,1] 
end

function ∇b(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    W1, b1, W2, b2 = flow.para_list
    y1 = W1*x .+ b1
    z1 = flow.σderi.(y1)

    grad = @tullio grad[i,k,a] := W2[i,j]*z1[j,a]*W1[j,k]
    return grad
end

function ∇b(flow::DynNNGenericTwo{T}, x::Vector{T}) where T <: AbstractFloat
    return ∇b(flow, reshape(x,(length(x),1)))[:,:,1]
end

function divg_b(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    W1, b1, W2, b2 = flow.para_list
    y1 = W1*x .+ b1
    z1 = flow.σderi.(y1)

    grad = @tullio grad[a] := W2[i,j]*z1[j,a]*W1[j,i]
    return grad
end

function divg_b(flow::DynNNGenericTwo{T}, x::Vector{T}) where T <: AbstractFloat
   return divg_b(flow, reshape(x,(length(x),1)))[1]
end

function grad_divg_b(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    W1, b1, W2, b2 = flow.para_list
    y1 = W1*x .+ b1
    z2 = flow.σsec_deri.(y1)
    
    grad = @tullio grad[k,a] := W2[i,j]*W1[j,i]*W1[j,k]*z2[j,a]
    return grad
end

function grad_divg_b(flow::DynNNGenericTwo{T}, x::Vector{T}) where T <: AbstractFloat
    return  grad_divg_b(flow, reshape(x,(length(x),1)))[:,1]
end

function grad_divg_wrt_para_part(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    num_particle = size(x,2)
    n = flow.dim
    m = flow.m
    
    W1, b1, W2, b2 = flow.para_list
    y1 = W1*x .+ b1
    z1 = flow.σderi.(y1)
    z2 = flow.σsec_deri.(y1)
    
    Fb2 = zeros(T,n,num_particle)
    
    Fb1 = @tullio Fb1[k,a] := W2[i,k]*W1[k,i]*z2[k,a]
    
    Fw1_part1 = @tullio Fw1_part1[l,θ,a] := W2[θ,l]*z1[l,a]
    Fw1_part2 = @tullio Fw1_part2[l,θ,a] := W2[i,l]*z2[l,a]*W1[l,i]*x[θ,a]
    Fw1 = Fw1_part1 + Fw1_part2
    # todo: why the following version failed?
    #Fw1 = @tullio Fw1[l,θ,a] := W2[θ,l]*z1[l,a] + W2[i,l]*z2[l,a]*W1[l,i]*x[θ,a]
    Fw1 = reshape(Fw1, (n*m,num_particle))
    
    Fw2 = @tullio Fw2[i,j,a] := z1[j,a]*W1[j,i]
    Fw2 = reshape(Fw2, (n*m, num_particle))
    
    return Fw1, Fb1, Fw2, Fb2
end

function grad_divg_wrt_para_part(flow::DynNNGenericTwo{T}, x::Vector{T}) where T<:AbstractFloat
    Fw1, Fb1, Fw2, Fb2 = grad_divg_wrt_para_part(flow, reshape(x,(length(x),1)))
    return Fw1[:,:,1], Fb1[:,1], Fw2[:,:,1], Fb2[:,1]
end

function grad_divg_wrt_para(flow::DynNNGenericTwo{T}, x::Matrix{T}) where T <: AbstractFloat
    num_particle = size(x,2)
    A = zeros(T, flow.total_num_para, num_particle)
    n = flow.dim
    m = flow.m
    l1 = n*m
    l2 = l1 + m
    l3 = l2 + n*m
    l4 = l3 + n
    Fw1, Fb1, Fw2, _ = grad_divg_wrt_para_part(flow, x)
    A[1:l1,:] .+= Fw1
    A[(l1+1):l2,:] .+= Fb1
    A[(l2+1):l3,:] .+= Fw2
    #A[(l3+1):l4,:] .+= Fb2 # Fb2 is always zero.
    return A
end

function grad_divg_wrt_para(flow::DynNNGenericTwo{T}, x::Vector{T}) where T <: AbstractFloat
   return grad_divg_wrt_para(flow, reshape(x, (length(x),1)))[:,1] 
end
