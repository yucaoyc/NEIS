export update_flow_para!, reshape_deri, update_flow!, get_ansatz_name

"""
Given a vector containing the direction that parameters need to move,
we update the parameters in the flow.
"""
function update_flow_para!(flow::DynTrain{T},
        vec_deri::Vector{}, h::T) where T<:AbstractFloat
    for i = 1:length(flow.num_para)
        flow.para_list[i] -= h*vec_deri[i]
    end
end

"""
Given a vector storing the derivative in fst_m
return an array that matches the size of all parameters.
"""
function reshape_deri(flow::DynTrain{T}, fst_m::Vector{T}) where T<:AbstractFloat

    num_of_para = length(flow.num_para)
    vec_deri = Vector{AbstractArray{T}}(undef, num_of_para)

    left_idx = 0
    for i = 1:num_of_para
        s = size(flow.para_list[i])
        L = prod(s)
        vec_deri[i] = reshape(fst_m[(left_idx+1):(left_idx+L)], s)
        left_idx += L
    end
    return vec_deri
end

function update_flow!(flow::DynTrain{T}, fst_m::Vector{T}, h::T) where T<:AbstractFloat
    vec_deri = reshape_deri(flow, fst_m)
    update_flow_para!(flow, vec_deri, h)
end

function get_ansatz_name(i::Int)
    if i == 0
        return "Linear ansatz"
    elseif i == 1
        return "Generic ansatz"
    elseif i == 2
        return "Gradient ansatz"
    elseif i == 4
        return "Simplified ansatz"
    end
end
