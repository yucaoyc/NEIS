using Test

export test_zero_var_dyn_for_infty_time

function test_zero_var_dyn_for_infty_time(U₀::Potential{T}, U₁::Potential{T},
        gpts::AbstractMatrix{T}, flow::Dyn{T};
        exact_mean::Union{T,Nothing}=nothing,
        ρdiff::Any=nothing, Nt::Int=200, ϵ::Number=0.02, solver=RK4) where T<:AbstractFloat

    if ρdiff == nothing
        test_func = [(x,t,p)->-U₀(x), (x,t,p)->-U₁(x), (x,t,p)-> divg_b(p, x)]
    else
        # if ∇ ⋅ 𝐛 = ρdiff
        test_func = [(x,t,p)->-U₀(x), (x,t,p)->-U₁(x), (x,t,p)-> ρdiff(x)]
    end
    test_func_para = [nothing nothing flow]

    offset = Int64(ceil(Nt/2))
    numsample = size(gpts,2)
    init_func = j -> gpts[:,j]

    function get_data(j)
        v, F₀, F₁, _ = one_path_dyn_b(flow, test_func, test_func_para, Nt, offset, init_func, j, solver)
        return [v, sum(F₁)/sum(F₀)]
    end

    data = zeros(T, 2, numsample)
    Threads.@threads for j = 1:numsample
        data[:,j] = get_data(j)
    end

    if exact_mean == nothing
        exact_mean = median(data[2,:])
    end
    @testset "test-zero-variance dynamics" begin
        v1 = abs(maximum(data[1,:])/exact_mean - 1.0)
        v2 = abs(minimum(data[1,:])/exact_mean - 1.0)
        v3 = abs(maximum(data[2,:])/exact_mean - 1.0)
        v4 = abs(minimum(data[2,:])/exact_mean - 1.0)
        @test maximum([v1,v2,v3,v4]) < ϵ
    end
end
