export load_eg1,
       load_eg2,
       load_eg3

function load_eg1(λ::T; n::Int=2, #λ::T=T(5.0),
        σsq::T=T(0.1),
        weight::Array{T}=T.([0.2, 0.8]),
        count_mode=:safe_count) where T <: AbstractFloat

    μ₀ = zeros(T,n)
    Σ₀ = Diagonal(ones(T,n))
    U₀ = Gaussian(n, μ₀, T(1.0))

    μlist = [T.([λ,0]),T.([0.0,-λ])]
    Σlist = [Diagonal(σsq*ones(T,n)) for i = 1:2]
    U₁ = generate_mixGaussian(n,μlist,Σlist,weight,count_mode=count_mode)

    exact_mean = dot(weight, [sqrt(det(σ)) for σ in Σlist])/sqrt(det(Σ₀))

    vanilla_m, _, vanilla_var = mean_msec_var(weight, μlist, Σlist)
    return U₀, U₁, exact_mean, vanilla_var/vanilla_m^2
end

# Todo: The implementation for eg2 can be further accelerated (a very minor thing to do though).
function load_eg2(n::Int, λ::T; σsq₁::T=T(0.1), σsq₂::T=T(0.5), num_pts=4,
        count_mode=:safe_count) where T<:AbstractFloat

    center_pts = [Array{T}([]) for i = 1:num_pts]
    for i = 1:num_pts
        θ = i*(2*pi/num_pts)
        center_pts[i] = T.(vcat(λ*[cos(θ), sin(θ)], zeros(T,n-2)))
    end

    μ₀ = zeros(T,n)
    Σ₀ = Diagonal(ones(T,n))
    U₀ = Gaussian(n, μ₀, T(1.0))

    weight = ones(T,num_pts)*(1/σsq₁/σsq₂^(T(n/2-1)))/num_pts
    Σlist = [Diagonal(vcat([σsq₁,σsq₁], σsq₂*ones(T,n-2))) for i = 1:num_pts]
    U₁ = generate_mixGaussian(n, center_pts, Σlist, weight, count_mode=count_mode)

    exact_mean = T(1.0)

    exact_mean_2 = dot(weight, [sqrt(det(σ)) for σ in Σlist])/sqrt(det(Σ₀))
    if abs(exact_mean - exact_mean_2) > 1.0e-4
        error("Incorrect value in eg2!")
    end

    vanilla_m, _, vanilla_var = mean_msec_var(weight, center_pts, Σlist)
    return U₀, U₁, exact_mean, vanilla_var/vanilla_m^2
end

"""
Correct the exact value
Count the effect of domain into consideration
For the Neal's funnel (10D) only.
"""
function partition_reduction_percent(Ωq::Function,
        numsample_domain::Int, n::Int, σf::T) where T<:AbstractFloat

    v = zeros(T,numsample_domain)
    for i = 1:numsample_domain
        x0 = randn(T)*σf
        x1 = randn(T,n-1)*exp(x0/2)
        v[i] = Ωq(vcat(x0, x1))
    end
    return sum(v)/numsample_domain
end

function load_eg3(n::Int, σf::T; σ₀::T=T(1.0), radius=T(25.0),
        count_mode=:safe_count) where T<:AbstractFloat
    Ωq(x) = domain_ball(x, radius)
    U₀ = Gaussian(n, zeros(T,n), σ₀^2)
    UU₁ = Funnel(n, σf)
    exact_mean = σf/σ₀^n
    U₁ = RestrictedPotential(n, UU₁, Ωq, count_mode=count_mode)

    # correct exact value due to restricted domain.
    percent_vec = map((j)->partition_reduction_percent(Ωq, 10^6, n, σf), 1:10)
    if std(percent_vec)/mean(percent_vec) < 1.0e-3
        reduce_percent = mean(percent_vec)
    else
        error("Not accurate enough estimates of the exact value.")
    end
    exact_mean *= reduce_percent

    return U₀, U₁, exact_mean, reduce_percent
end
