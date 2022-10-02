using Test

export test_zero_var_dyn_for_infty_time

"""
This function tests whether the flow is a zero-variance dynamics using NEIS for U₀ and U₁.
- gpts contains points to test whether H(x) := ∫ exp(-U₁(Xₜ(x))) Jₜ(x) dt / ∫ exp(-U₀(Xₜ(x))) Jₜ(x) dt = Z₁/Z₀.
- If exact_mean ≡ Z₁/Z₀ is available, we use it as a reference; otherwise, we only test
if the random variable H(x), x∼ρ₀ has well-concentrated values.
- When ∇ ⋅ 𝐛 = ρdiff is available, we can simply pass it for better performance.
- Nt is the number of time grid points.
- ϵ is the relative error tolerance.
- solver is the ODE solver.

Remark: Here we use the finite-time NEIS scheme with T₋ = -1/2.
Therefore, the flow needs to be already rescaled before using this function.
"""
function test_zero_var_dyn_for_infty_time(U₀::Potential{T}, U₁::Potential{T},
        gpts::AbstractMatrix{T}, flow::Dyn{T};
        exact_mean::Union{T,Nothing}=nothing,
        ρdiff::Any=nothing, Nt::Int=10^3, ϵ::Number=0.02, solver=RK4) where T<:AbstractFloat

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


"""
This function tests whether the flow is a zero-variance dynamics using NEIS for U₀ and U₁.
- gpts contains points to test whether H(x) := ∫ exp(-U₁(Xₜ(x))) Jₜ(x) dt / ∫ exp(-U₀(Xₜ(x))) Jₜ(x) dt = Z₁/Z₀.
- If exact_mean ≡ Z₁/Z₀ is available, we use it as a reference; otherwise, we only test
if the random variable H(x), x∼ρ₀ has well-concentrated values.
- Tscale is the truncation time to estimate the above integral.
- When ∇ ⋅ 𝐛 = ρdiff is available, we can simply pass it for better performance.
- Nt is the number of time grid points.
- ϵ is the relative error tolerance.
- solver is the ODE solver.

Remark: Here we use the finite-time NEIS scheme with T₋ = -1/2.
The flow will be rescaled by Tscale.
"""
function test_zero_var_dyn_for_infty_time(U₀::Potential{T}, U₁::Potential{T},
        gpts::AbstractMatrix{T}, flow_original::Dyn{T}, Tscale::T;
        exact_mean::Union{T,Nothing}=nothing,
        ρdiff::Any=nothing, Nt::Int=10^3, ϵ::Number=0.02, solver=RK4) where T<:AbstractFloat

    flow = DynFix{T}(flow_original.dim, x-> Tscale*flow_original(x))
    if ρdiff != nothing
        ρdiff_scaled = x->Tscale*ρdiff(x)
    else
        ρdiff_scaled = nothing
    end
    return test_zero_var_dyn_for_infty_time(U₀, U₁, gpts, flow, exact_mean = exact_mean,
                                            ρdiff = ρdiff_scaled, Nt = Nt, ϵ = ϵ, solver = solver)

end
