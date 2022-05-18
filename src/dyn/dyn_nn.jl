using Flux
using Flux.Optimise: update!
using ForwardDiff
using LinearAlgebra

export DynNN,
    vectorize, # utility functions
    sigmoid_deri,
    update_flow_para!,
    reshape_deri,
    ∇b, # general functions
    divg_b,
    grad_divg_b,
    grad_b_wrt_para,
    grad_divg_wrt_para,
    grad_V_wrt_para,
    ∇b_old, # reference implementations
    divg_b_old,
    grad_b_wrt_para_old,
    grad_b_wrt_para_unstable,
    divg,
    grad_divg_wrt_para_old


mutable struct DynNN <: Dyn
    dim::Int64 # dimension
    model # nn representation for dynamics B: R^d -> R^d.
    para_list::Array{} # an array for (W_1, b_1, W_2, b_2 ..., W_L, b_L)
    train_para_idx::Array{Int64} # indices for the parameters in para_list to train (also included in p)
    f # A function that take x, W_1, b_1, W_2, b_2, ⋯, W_L, b_L as inputs.
    dyn_type
    num_para::Array{Int64}
    total_num_para::Int64
    J::VecType
    σ
    σderi
    σsec_deri
end

function DynNN(dim, model, para_list, train_para_idx, f, dyn_type)
    DynNN(dim, model, para_list, train_para_idx, f, dyn_type, nothing, nothing, nothing)
end

function DynNN(dim, model, para_list, train_para_idx, f, dyn_type, σ, σderi, σsec_deri)
    num_para = zeros(Int64, length(train_para_idx))
    for i = 1:length(num_para)
        num_para[i] = length(para_list[train_para_idx[i]])
    end
    total_num_para = sum(num_para)
    return DynNN(dim, model, para_list, train_para_idx,
                 f, dyn_type,
                 num_para, total_num_para, [], σ, σderi, σsec_deri)
end

#######################################################
# Some unitity functions
#######################################################

vectorize(a::Union{Matrix,CuArray}) = reshape(a,length(a))
vectorize(a::Array{}) = a

sigmoid_deri(x) = exp(-max(x,-200))*sigmoid(x)^2

function update_flow_para!(flow, vec_deri::VecType, num_of_para::Int64, h::Float64)
    for i = 1:num_of_para
        p_idx = flow.train_para_idx[i]
        flow.para_list[p_idx] -= h*vec_deri[i]
    end
end

"""
    given a vector storing the derivative in fst_m
    return an array that matches the size of all parameters
"""
function reshape_deri(flow, fst_m::VecType)

    num_of_para = length(flow.train_para_idx)
    vec_deri = Array{VecType}(undef, num_of_para)

    left_idx = 0
    for i = 1:num_of_para
        p_idx = flow.train_para_idx[i]
        s = size(flow.para_list[p_idx])
        L = prod(s)
        vec_deri[i] = reshape(fst_m[(left_idx+1):(left_idx+L)], s)
        left_idx += L
    end

    return vec_deri
end

function ∇b(b::DynNN, x::VecType)
    if length(b.para_list) == 4
        if b.dyn_type == 1
            return ∇b_nn2(b.σderi, x, b.para_list...)
        elseif b.dyn_type == 2
            return ∇b_nn2_scalar(b.σsec_deri, x, b.para_list...)
        elseif b.dyn_type == 3
            return ∇b_nn2_scalarJ(b.J, b.σsec_deri, x, b.para_list...)
        end
    end

    if length(b.para_list) == 6
        if b.dyn_type == 1
            return ∇b_nn3(b.σ, b.σderi, x, b.para_list...)
        elseif b.dyn_type == 2
            return ∇b_nn3_scalar(b.σ, b.σderi, b.σsec_deri, x, b.para_list...)
        elseif b.dyn_type == 3
            return ∇b_nn3_scalarJ(b.J, b.σ, b.σderi, b.σsec_deri, x, b.para_list...)
        end
    end

    return ∇b_old(b, x)
end

function divg_b(b::DynNN, x::VecType)
    if b.dyn_type == 3
        return 0.0
    else
        return tr(∇b(b, x))
    end
end

function grad_divg_b(b::DynNN, x::VecType)
    if b.dyn_type != 3
        f(x) =  divg_b(b, x)
        return ForwardDiff.gradient(f, x)
    else
        return zeros(b.dim)
    end
end

function grad_b_wrt_para(b::DynNN, x::VecType)
    if length(b.para_list) <= 6
        return grad_b_wrt_para_unstable(b, x)
    else
        return grad_b_wrt_para_old(b, x)
    end
end

function grad_divg_wrt_para(b::DynNN, x::VecType)
    if b.dyn_type == 3
        return zeros(b.total_num_para)
    else
        if length(b.para_list) == 4 # a two-layer nn
            return grad_divg_wrt_para_nn2(b.σ, b.σderi, b.σsec_deri, b, x)
        elseif length(b.para_list) == 6
            return grad_divg_wrt_para_nn3(b.σ, b.σderi, b.σsec_deri, b, x)
        else
            return grad_divg_wrt_para_old(b, x)
        end
    end
end


"""
Compute the gradient of a potential function V wrt trainnable parameters.

Caution:
* if V is implemented with automatic differentiation, this function might not work.
"""
function grad_V_wrt_para(V::DynNN, x::VecType)

    grad1 = Flux.jacobian((z...)->V.f(x,z...), V.para_list...)

    total_num_para = V.total_num_para
    num_para = V.num_para
    grad = zeros(total_num_para)
    count = 0
    for i = 1:length(num_para)
        p_idx = V.train_para_idx[i]
        new_count = count + num_para[i]
        grad[(count+1):(new_count)] = grad1[p_idx]
        count = new_count
    end
    return grad

end

###########################################
# auxillary functions herein.
###########################################

function ∇b_old(b::DynNN, x::VecType)
    ForwardDiff.jacobian(z -> b.f(z, b.para_list...), x)
end

function divg_b_old(b::DynNN, x::VecType)
    tr(ForwardDiff.jacobian(z->b.f(z, b.para_list...), x))
end

function grad_b_wrt_para(b::DynNN, x::VecType, i::Int64)
    ℓ = length(b.para_list)
    h = z -> b.f(x, b.para_list[1:(i-1)]..., z, b.para_list[(i+1):ℓ]...)
    return ForwardDiff.jacobian(h, b.para_list[i])
end

function divg(b::DynNN, x::VecType, args...)
    tr(ForwardDiff.jacobian(z -> b.f(z, args...), x))
end

function grad_divg_wrt_para(b::DynNN, x::VecType, i::Int64)

    ℓ = length(b.para_list)
    if i < 0 || i > ℓ
        @error "index cannot exceed ℓ nor negative."
        return nothing
    end

    if b.dyn_type != 3
        h = z -> divg(b, x, b.para_list[1:(i-1)]..., z, b.para_list[(i+1):ℓ]...)
        return vectorize(ForwardDiff.gradient(h, b.para_list[i]))
    else
        return zeros(length(b.para_list[i]))
    end
end

function grad_b_wrt_para_old(b::DynNN, x::VecType)
    # compute ∂_θ b(x, θ)
    total_num_para = b.total_num_para
    num_para = b.num_para

    grad = zeros(b.dim, total_num_para)
    count = 0
    for i = 1:length(num_para)
        p_idx = b.train_para_idx[i]
        new_count = count + num_para[i]
        grad[:,(count+1):(new_count)] = grad_b_wrt_para(b, x, p_idx)
        count = new_count
    end
    return grad
end

function grad_b_wrt_para_unstable(b::DynNN, x::VecType)

    grad1 = Flux.jacobian((z...)->b.f(x,z...), b.para_list...)

    total_num_para = b.total_num_para
    num_para = b.num_para
    grad = zeros(b.dim, total_num_para)
    count = 0
    for i = 1:length(num_para)
        p_idx = b.train_para_idx[i]
        new_count = count + num_para[i]
        grad[:,(count+1):(new_count)] = grad1[p_idx]
        count = new_count
    end
    return grad
end

function grad_divg_wrt_para_old(b::DynNN, x::VecType)
    # compute ∇_θ (∇⋅b)
    total_num_para = b.total_num_para
    num_para = b.num_para

    grad = zeros(total_num_para)
    count = 0
    for i = 1:length(num_para)
        p_idx = b.train_para_idx[i]
        new_count = count + num_para[i]
        grad[(count+1):(new_count)] = grad_divg_wrt_para(b, x, p_idx)
        count = new_count
    end
    return grad
end
