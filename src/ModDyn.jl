module ModDyn

using CUDA
using ModPotential

export Dyn, use_gpu, 
    gpuarray, gpulayer, 
    gpurandn, gpuzeros,
    VecType

abstract type Dyn end

#global use_gpu = true
global use_gpu = false

if CUDA.functional() & use_gpu
    function gpuarray(x) 
        return CuArray(x) 
    end
    function gpulayer(x)
        println("Convert layer to GPU")
        return fmap(cu,x) 
    end
    function gpurandn(n)
        CUDA.randn(n) 
    end
    function gpuzeros(n)
        CUDA.fill(0.0f0,n)
    end
else
    function gpuarray(x) return x end
    function gpulayer(x) return x end
    function gpurandn(n) return randn(n) end
    function gpuzeros(n) return zeros(n) end
end

VecType=Union{Array,CuArray}

include("dyn/dyn_nn.jl")
include("dyn/dynfix.jl") # a fixed dynamics
include("dyn/grad_nn.jl")
include("dyn/init_nn.jl")
include("dyn/ode_flow.jl")
include("dyn/dyn_nn_rescale.jl")
include("dyn/dyn_ul.jl")
include("dyn/dyn_funnelexpansatz.jl")

end
