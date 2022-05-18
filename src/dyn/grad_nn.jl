###############################################################################
# Specialized functions for nn2
##############################################################################

function ∇b_nn2(σderi, x, args...)
    interm = args[1]*x .+ args[2]
    args[3]*Diagonal(σderi.(interm))*args[1]
end

function b_nn2_scalar(σderi, x, args...)
    # V is a two-layer nn with output dimn = 1
    # return ∇V
    interm = args[1]*x .+ args[2]
    return args[1]'*(args[3][:].*σderi.(interm))
end

function ∇b_nn2_scalar(σsec_deri, x, args...)
    # V is a two-layer nn with output dimn = 1
    # return ∇²V (Hessian)
    interm = args[1]*x .+ args[2]
    return (args[1]')*Diagonal(args[3][:].*σsec_deri.(interm))*args[1]
end

function b_nn2_scalarJ(J::VecType, σderi, x, args...)
    return J*b_nn2_scalar(σderi, x, args...)
end

function ∇b_nn2_scalarJ(J::VecType, σsec_deri, x, args...)
    return J*∇b_nn2_scalar(σsec_deri, x, args...)
end

function grad_divg_wrt_para_nn2(σ, σderi, σsec_deri, b::DynNN, x::VecType)
    if b.dyn_type == 1
        grad1 = Flux.gradient((z...)->tr(∇b_nn2(σderi, x, z...)), b.para_list...)
    elseif b.dyn_type == 2
        grad1 = Flux.gradient((z...)->tr(∇b_nn2_scalar(σsec_deri, x, z...)), b.para_list...)  
    end
    return transform_grad_divg_wrt_para(grad1, b)
end

function transform_grad_divg_wrt_para(grad1, b::DynNN)
    total_num_para = b.total_num_para
    num_para = b.num_para
    grad = zeros(total_num_para)
    count = 0
    for i = 1:length(num_para)
        p_idx = b.train_para_idx[i]
        new_count = count + num_para[i]
        if grad1[p_idx] != nothing
            grad[(count+1):(new_count)] = vectorize(grad1[p_idx])
        end
        count = new_count
    end
    return grad
end

######################################################################
function ∇b_nn3(σ, σderi, x, args...)
    x1 = args[1]*x .+ args[2]
    x2 = args[3]*σ.(x1) .+ args[4]
    args[5]*Diagonal(σderi.(x2))*args[3]*Diagonal(σderi.(x1))*args[1]
end

function grad_divg_wrt_para_nn3(σ, σderi, σsec_deri, b::DynNN, x::VecType)
    if b.dyn_type == 1
        grad1 = Flux.gradient((z...)->tr(∇b_nn3(σ, σderi, x, z...)), b.para_list...)
    elseif b.dyn_type == 2
    	grad1 = Flux.gradient((z...)->tr(∇b_nn3_scalar(σ, σderi, σsec_deri, x, z...)), b.para_list...)
    end
    
	return transform_grad_divg_wrt_para(grad1, b)
end

function b_nn3_scalar(σ, σderi, x::VecType, args...)
    x₁ = args[1]*x .+ args[2]
    x₂ = args[3]*σ.(x₁) .+ args[4]
    ∇x₁ = args[1]
    ∇x₂ = args[3]*Diagonal(σderi.(x₁))*∇x₁
    return ∇x₂'*(args[5][:].*σderi.(x₂)) 
end

function ∇b_nn3_scalar(σ, σderi, σsec_deri, x, args...)
    x₁ = args[1]*x .+ args[2]
    x₂ = args[3]*σ.(x₁) .+ args[4]
    ∇x₁ = args[1]
    ∇x₂ = args[3]*Diagonal(σderi.(x₁))*∇x₁
    
    tmp = args[5]*Diagonal(σderi.(x₂))*args[3]
    return ∇x₂'*Diagonal(args[5][:].*σsec_deri.(x₂))*∇x₂ + ∇x₁'*(Diagonal(tmp[:].*σsec_deri.(x₁)))*∇x₁
end

function b_nn3_scalarJ(J::VecType, σ, σderi, x, args...)
    return J*b_nn3_scalar(σ, σderi, x, args...)
end

function ∇b_nn3_scalarJ(J::VecType, σ, σderi, σsec_deri, x, args...)
    return J*∇b_nn3_scalar(σ, σderi, σsec_deri, x, args...)
end
