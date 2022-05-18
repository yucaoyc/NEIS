export f₀, # initialization
    init_dyn_nn_general,
    init_dyn_nn_gradient,
    init_dyn_nn_divfree,
    init_dyn_nn

function nn_1(x, W1, b1, activation)
   return activation.(W1*x.+b1)
end

function nn_2(x, W1, b1, W2, b2, activation)
    return activation.(W2*nn_1(x,W1,b1,activation).+b2)
end

function nn_3(x, W1, b1, W2, b2, W3, b3, activation)
    return activation.(W3*nn_2(x, W1, b1, W2, b2, activation).+b3)
end

function nn_template(x, args...; activation=softplus)
    L = length(args)
    if L == 2
        return nn_1(x,args[1],args[2],activation)
    elseif L == 4
        return nn_2(x,args[1],args[2],args[3],args[4],activation)
    elseif L == 6
        return nn_3(x,args[1],args[2],args[3],args[4],args[5],args[6],activation)
    else
        return activation.(args[L-1]*nn_template(x, args[1:(L-2)]...; activation=activation) .+ args[L])
    end
end

function f₀(x, args...; activation=softplus) 
    L = length(args); 
    return args[L-1]*nn_template(x, args[1:(L-2)]...; activation=activation) .+ args[L]
end

function init_dyn_nn_general(ndims::Array{Int64};  
        activation=softplus, σderi=sigmoid, σsec_deri = sigmoid_deri, 
        scale=1.0, init = Flux.glorot_uniform, to_init_b=true,
        to_train_bL=false, to_init_bL=false)

    if to_init_bL == true && to_train_bL == false
       @warn "b_L is initialized but not trainable! Make sure this is what you want." 
    end

    n = ndims[1]
    ℓ = length(ndims) - 1 # layer number
    if ℓ < 2
        error("The layer length must be at least 2.")
        return
    end
    
    lay = []
    para_list = Array{Any}(undef,0)
    for i = 1:ℓ
        if i != ℓ
            push!(lay, gpulayer(f64(Dense(ndims[i], ndims[i+1], activation, init = init))))
        else
            push!(lay, gpulayer(f64(Dense(ndims[i], ndims[i+1], init = init))))
        end
        if to_init_b && i != ℓ
            δb = gpuarray(randn(size(lay[i].b))); 
            δb ./= norm(δb)
            update!(lay[i].b, lay[i].b .- δb)
        end
        if to_init_bL && i == ℓ
            δb = gpuarray(randn(size(lay[i].b))); 
            δb ./= norm(δb)
            update!(lay[i].b, lay[i].b .- δb)
        end
        push!(para_list, lay[i].W); 
        push!(para_list, lay[i].b)
    end
    lay[ℓ].W .*= scale
    lay[ℓ].b .*= scale

    model = Chain(lay...)

    if to_train_bL == false
        train_para_idx = Array(1:(2*ℓ-1))
    else
        train_para_idx = Array(1:(2*ℓ))
    end
    
    if ℓ == 2
        f = (x,W1,b1,W2,b2) -> W2*nn_1(x,W1,b1,activation) .+ b2
    elseif ℓ == 3
        f = (x,W1,b1,W2,b2,W3,b3) -> W3*nn_2(x,W1,b1,W2,b2,activation).+b3
    elseif ℓ == 4
        f = (x,W1,b1,W2,b2,W3,b3,W4,b4) -> W4*nn_3(x,W1,b1,W2,b2,W3,b3,activation).+b4
    else # ℓ >= 5
        f(args...) = f₀(args...; activation=activation)
    end
    
    return DynNN(n, model, para_list, train_para_idx, f, 1, activation, σderi, σsec_deri)

end

function init_dyn_nn_gradient(ndims::Array{Int64};  
        activation=softplus, σderi=sigmoid, σsec_deri=sigmoid_deri, 
        scale=1.0, init = Flux.glorot_uniform, to_init_b=true)
    
    n = ndims[1]
    ℓ = length(ndims) - 1 # layer number
    
    ℓ < 2 ? error("The layer length must be at least 2.") : nothing
    (ndims[ℓ+1] != 1) ? error("The last entry in ndims must be 1.") : nothing
    
    V = init_dyn_nn_general(ndims; activation=activation,
            σderi=σderi, σsec_deri=σsec_deri, 
            scale=scale, init = init, to_init_b=to_init_b)
    
    model₀(x) = V.model(x)[1]
    model(x) = ForwardDiff.gradient(model₀, x)

    function f(x, args...)
        if ℓ == 2
            # a specialized function for layer 2 for speed-up.
            return b_nn2_scalar(σderi, x, args...)
        elseif ℓ == 3
            return b_nn3_scalar(activation, σderi, x, args...)
        else
            return ForwardDiff.gradient(z -> V.f(z,args...)[1], x)
        end
    end
   
    return DynNN(n, model, V.para_list, V.train_para_idx, f, 2, activation, σderi, σsec_deri), V
end


function init_dyn_nn_divfree(ndims::Array{Int64};  
        activation=softplus, σderi=sigmoid, σsec_deri = sigmoid_deri, 
        scale=1.0, init = Flux.glorot_uniform, to_init_b=true)
    
    n = ndims[1]
    ℓ = length(ndims) - 1 # layer number
    
    ℓ < 2 ? error("The layer length must be at least 2.") : nothing
    (ndims[ℓ+1] != 1) ? error("The last entry in ndims must be 1.") : nothing
    mod(n, 2) != 0 ? error("Input dimension n must be even!") : nothing
    
    d = Int64(ceil(n/2))
    J = zeros(n,n)
    J[1:d, (d+1):(2*d)] = Matrix(1.0I,d,d)
    J[(d+1):(2*d),1:d] = -Matrix(1.0I,d,d)
    J = gpuarray(J)

    V = init_dyn_nn_general(ndims; activation=activation, 
            σderi=σderi, σsec_deri=σsec_deri, 
            scale=scale, init = init, to_init_b=to_init_b)
    
    model₀(x) = V.model(x)[1]
    model(x) = J*ForwardDiff.gradient(model₀, x)
    function f(x, args...) 
       if ℓ == 2
           return b_nn2_scalarJ(J, σderi, x, args...)
       elseif ℓ == 3
           return b_nn3_scalarJ(J, activation, σderi, x, args...) 
        else
            return J*ForwardDiff.gradient(z -> V.f(z,args...)[1], x)
        end
    end
    b = DynNN(n, model, V.para_list, V.train_para_idx, f, 3, activation, σderi, σsec_deri)
    b.J = J

    return b, V
end

function init_dyn_nn(ndims::Array{Int64}, model_idx::Int64;
        activation=softplus, σderi=sigmoid, σsec_deri = sigmoid_deri, 
        scale=1.0, init = Flux.glorot_uniform, to_init_b=true,
        to_train_bL=false, to_init_bL=false)

    if model_idx == 1
        b = init_dyn_nn_general(ndims; 
                activation=activation, σderi=sigmoid, σsec_deri = sigmoid_deri, 
                scale=scale, init=init, to_init_b = to_init_b,
                to_train_bL = to_train_bL, to_init_bL = to_init_bL);
        V = nothing
    elseif model_idx == 2
        new_ndims = copy(ndims)
        new_ndims[length(new_ndims)] = 1
        b, V = init_dyn_nn_gradient(new_ndims;
                activation=activation, σderi=sigmoid, σsec_deri = sigmoid_deri, 
                scale=scale, init=init, to_init_b = to_init_b);
    else
        new_ndims = copy(ndims)
        new_ndims[length(new_ndims)] = 1
        b, V = init_dyn_nn_divfree(new_ndims;
                activation=activation, σderi=sigmoid, σsec_deri = sigmoid_deri, 
                scale=scale, init=init, to_init_b = to_init_b);
    end
    return b, V

end
