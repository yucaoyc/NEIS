export DynNNRescale, convert_DynNN_with_rescale

"""
    the dynamics is 
        b(x) = α(x) * flow(x)
    where flow is DynNN.
"""
mutable struct DynNNRescale <: Dyn
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
    flow::DynNN
    α
    αgrad
    αhess
end

function convert_DynNN_with_rescale(flow::DynNN, α, αgrad, αhess)
    f(x, args...) = flow.f(x, args...)*α(x)
    model(x) = f(x, flow.para_list...)
    
    return DynNNRescale(flow.dim, model, flow.para_list, flow.train_para_idx, 
        f, flow.dyn_type, flow.num_para, flow.total_num_para, 
        flow.J, flow.σ, flow.σderi, flow.σsec_deri, flow, α, αgrad, αhess)
end

function ∇b(b::DynNNRescale, x::Union{VecType,Array})
    return b.flow.f(x, b.flow.para_list...)*b.αgrad(x)' + b.α(x)*∇b(b.flow, x)
end

function divg_b(b::DynNNRescale, x::VecType)
    return b.αgrad(x)'*b.flow.f(x, b.flow.para_list...) + b.α(x)*divg_b(b.flow, x)
end

function grad_divg_b(b::DynNNRescale, x::VecType)
    v = b.αhess(x)' * b.flow.f(x, b.flow.para_list...)
    v = v + ∇b(b.flow, x)' * b.αgrad(x)
    v = v + b.αgrad(x) * divg_b(b.flow, x)
    v = v + b.α(x) * grad_divg_b(b.flow, x)
    return v
end

function grad_b_wrt_para(b::DynNNRescale, x::VecType)
    return b.α(x)*grad_b_wrt_para(b.flow, x)
end

function grad_b_wrt_para(b::DynNNRescale, x::VecType, i::Int64)
    return b.α(x)*grad_b_wrt_para(b.flow, x, i)
end

function grad_divg_wrt_para(b::DynNNRescale, x::VecType)
    return (b.αgrad(x)'*grad_b_wrt_para(b.flow, x))' + b.α(x)*grad_divg_wrt_para(b.flow, x)
end

function grad_divg_wrt_para(b::DynNNRescale, x::VecType, i::Int64)
    return (b.αgrad(x)'*grad_b_wrt_para(b.flow, x, i))' + b.α(x)*grad_divg_wrt_para(b.flow, x, i)
end
