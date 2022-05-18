module ModTestCase

using Printf, LinearAlgebra, Random

using ModPotential
using ModUtil: randHermitian
using ForwardDiff
using ModPoisson
using ModDyn
using Distributions

export load_testcase, 
    load_mixGaussian,
    load_two_mode_Gaussian_PoisDyn,
	load_funnel,
    load_cauchy

function load_testcase(testcasenum::Int64, seed::Int64, n::Int64, 
        λ::Float64, σmin::Float64, σmax::Float64; 
        weight::Array{}=[], reducedsysonly=true)

    folder = @sprintf("test_case_%d", testcasenum)

    Random.seed!(seed)

    μ₁ = zeros(n)
    Σ₁ = 1.0*Matrix(I,n,n)
    H₁ = generate_Gaussian_Potential(n,μ₁,Σ₁,func_sampler=(z)->randn(n));
    U₁ = generate_extended_Potential(H₁, 1.0)

    if testcasenum == 1
        #μ₂ = ones(n)*λ
        μ₂ = zeros(n)
        μ₂[1] = λ
        Σ₂ = randHermitian(n,σmin,σmax)
        H₂ = generate_Gaussian_Potential(n,μ₂,Σ₂)
        exact_mean = sqrt(det(Σ₂))/sqrt(det(Σ₁))
    elseif testcasenum == 2 && n == 2
        μlist = [[λ,0],[0.0,-λ]]
        Σlist = [randHermitian(n,σmin,σmax),randHermitian(n,σmin,σmax)]
        H₂ = generate_mixGaussian_Potential(n,μlist,Σlist,weight)
        exact_mean = dot(weight, [sqrt(det(σ)) for σ in Σlist])/sqrt(det(Σ₁))
    else
        error("Please use the correct testcasenum!")
    end        
    U₂ = generate_extended_Potential(H₂, 1.0)
   
    if reducedsysonly 
        return (H₁, H₂, exact_mean)
    else
        return (H₁, U₁, H₂, U₂, exact_mean, folder)
    end
end


function load_mixGaussian(n::Int64, σsq₁::Float64, σsq₂::Float64, center_points::Array{}; 
        σsq₀::Float64 = 1.0, reducedsysonly=true, version=2)

    σ₀ = sqrt(σsq₀)
#    if version == 2
        H₁ = FixedPotential(n, 
            x->norm(x)^2/(2*σsq₀), 
            x->x/σsq₀, 
            x->Matrix(1.0I,n,n)/σsq₀, 
            x->n/σsq₀,
            [], 
            (z)->σ₀*randn(n))
#    else
#        μ₁ = zeros(n)
#        Σ₁ = σsq₀*Matrix(I,n,n)
#        H₁ = generate_Gaussian_Potential(n, μ₁, Σ₁, func_sampler=(z)->σ₀*randn(n));
#    end
    U₁ = generate_extended_Potential(H₁, 1.0)

#    if version == 1
#        μlist = []
#        Σlist = []
#        for i = 1:length(center_points)
#            vec = zeros(n)
#            pt = center_points[i]
#            vec[1:2] = pt
#            push!(μlist,vec)
#            sigma_diag = vcat([σsq₁, σsq₁], σsq₂*ones(n-2))
#            push!(Σlist, Diagonal(sigma_diag))
#        end
#        weight = [1/sqrt(det(Σ))/length(center_points) for Σ in Σlist]
#        H₂ = generate_mixGaussian_Potential(n,μlist,Σlist,weight)
#        U₂ = generate_extended_Potential(H₂, 1.0)
#        exact_mean = dot(weight, [sqrt(det(Σ)) for Σ in Σlist])/sqrt(det(Σ₁))
#    else
        mode = length(center_points)
        Σlist = σsq₁*ones(mode)
        weight = ones(mode)*(1/σsq₁/σsq₂^(n/2-1))/mode
        H₂tmp = generate_mixGaussian_Potential(2, center_points, Σlist, weight)

        HessU(x) = vcat(hcat(H₂tmp.HessU(x[1:2]), zeros(2,n-2)),
                hcat(zeros(n-2,2), Matrix(I,n-2,n-2)/σsq₂))
        LaplaceU(x) = H₂tmp.LaplaceU(x[1:2]) + (n-2)/σsq₂
        
        H₂ = FixedPotential(n, x-> H₂tmp.U(x[1:2]) + norm(x[3:n])^2/(2*σsq₂),
                            x->vcat(H₂tmp.gradU(x[1:2]), x[3:n]/σsq₂), 
                            HessU, LaplaceU, [], nothing)
        U₂ = generate_extended_Potential(H₂, 1.0)
        exact_mean = 1.0
 #   end
   
    if reducedsysonly
        return (H₁, H₂, exact_mean)
    else
        return (H₁, U₁, H₂, U₂, exact_mean)
    end
end


function load_two_mode_Gaussian_PoisDyn(n, λ, σmin)

    center_pts = [[λ, 0.0], [0.0, -λ]]
    weight = [0.5, 0.5]
    U₀, U₁, exact_mean = load_mixGaussian(n, σmin, σmin, center_pts)
    
    Z₀ = sqrt(2*pi)^n

    function ρ₀(x::Array{})
        exp(-U₀.U(x))/Z₀
    end

    function ρ₁(x::Array{})
        exp(-U₁.U(x))/Z₀
    end

    function ρ₀(x::Float64,y::Float64)
        return exp(-U₀.U([x,y]))/Z₀
    end

    function ρ₁(x::Float64,y::Float64)
        exp(-U₁.U([x,y]))/Z₀
    end
    
    b = exact_dynb_poisson(n, [1.0], [[0.0, 0.0]], [1.0], weight, center_pts, [σmin, σmin])
    flow = DynNN(n, nothing, [], [], (x,args...)->b(x), :poisson);

    return U₀, U₁, exact_mean, ρ₀, ρ₁, b, flow

end

function load_funnel(n, σf; reducedsystemonly=true, σ₀ = 1.0)
    
    U₀ = generate_Gaussian_Potential(n, zeros(n), σ₀^2, func_sampler=(z)->σ₀*randn(n))
    Uext₀ = generate_extended_Potential(U₀, 1.0)
    U₁ = generate_Funnel_Potential(n, σf)
    Uext₁ = generate_extended_Potential(U₁, 1.0)
    exact_mean = σf/σ₀^n
    if reducedsystemonly
        return U₀, U₁, exact_mean
    else
        return U₀, Uext₀, U₁, Uext₁, exact_mean
    end
end

function load_cauchy(n, μ, σ)
    
    c1avg(z) = (cauchy(z,-μ,σ)+cauchy(z,μ,σ))/2
    c1deri(z) = (cauchyderi(z,-μ,σ)+cauchyderi(z,μ,σ))/2
    ϕ₁(z) = -log.(c1avg.(z))
    ψ₁(z) = -c1deri(z)/c1avg(z)
    U₁ = FixedPotential(n, x->sum(ϕ₁.(x)), x->map(ψ₁, x), nothing, nothing, [], nothing) 
    
    sampler_dist = Cauchy(0, σ)
    c0avg(z) = cauchy(z, 0, σ)
    c0deri(z) = cauchyderi(z, 0, σ)
    ϕ₀(z) =  -log.(c0avg.(z)) 
    ψ₀(z) = -c0deri(z)/c0avg(z) 
    U₀ = FixedPotential(n, x->sum(ϕ₀.(x)), x->map(ψ₀, x), nothing, nothing, [], j->rand(sampler_dist,n)) 
    
    exact_mean = 1.0
    
    return U₀, U₁, exact_mean
end

end
