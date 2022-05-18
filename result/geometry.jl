push!(LOAD_PATH,"../src")
using Plots
using ModUtil
plot_setup()
using LinearAlgebra
folder = "./assets/"
~isdir(folder) ? mkdir(folder) : nothing

function get_sample_points(radius, n; shift=0)
    """
        Generate sample points according to radius and number of points.
    """
    
    θ = range(0, 2*π, length=n+1)
    θ = θ[1:n] .+ shift
    return [radius*cos(α) for α in θ], [radius*sin(α) for α in θ]
end

function prob_isotropic_gaussian_wrt_radius(a, σ)
    """
        Probability of 2D gaussian within radius a.
    """
    
    return 1 - exp(-a^2/(2*σ^2))
end

function get_sample_points_isotropic_gaussian(μ, σ, scale, δr; num_round=3, to_shift=true)
    """
        Generate sample points for a 2D isotropic Gaussian with mean μ and std σ.
        scale controls the number of points
        δr gives the radius increment for sample points.
    """
    
    ncircle = Int64(ceil(2*σ/δr))
    x = []
    y = []
    for i = 1:ncircle
        prob = prob_isotropic_gaussian_wrt_radius(i*δr, σ) - prob_isotropic_gaussian_wrt_radius((i-1)*δr, σ)
        num_points = Int64(ceil(prob*scale))
        if to_shift 
            xtmp, ytmp = get_sample_points(i*δr, num_points, shift=i*num_round*2*π/ncircle)
        else
            xtmp, ytmp = get_sample_points(i*δr, num_points)
        end
        x = vcat(x, xtmp)
        y = vcat(y, ytmp)
    end
    return x .+ μ[1], y .+ μ[2]
end

function generate_samples_gaussian(scale, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax; figscale=80)

    figsize = (figscale*(xmax-xmin),figscale*(ymax-ymin))

    x0, y0 = get_sample_points_isotropic_gaussian([0.0, 0.0], σ₀, scale, δr, num_round = num_round)
    fig = scatter(x0, y0, label="", xlims=(xmin,xmax), ylims=(ymin,ymax), size=figsize)

    x1, y1 = get_sample_points_isotropic_gaussian([λ, 0.0], σ₁, scale, δr, num_round=num_round)
    scatter!(x1, y1, label="")
    
    return fig
    
end

function generate_samples_2mode_gaussian(scale, weight, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax; 
        figscale=80, to_shift=true)
    figsize = (figscale*(xmax-xmin),figscale*(ymax-ymin))
    
    x0, y0 = get_sample_points_isotropic_gaussian([0.0, 0.0], σ₀, scale, δr, 
        num_round = num_round, to_shift=to_shift)
    scatter(x0, y0, label="", xlims=(xmin,xmax), ylims=(ymin,ymax), size=figsize)

    x1, y1 = get_sample_points_isotropic_gaussian([λ, 0.0], σ₁, Int64(round(scale*weight[1])), δr, 
        num_round=num_round, to_shift=to_shift)
    scatter!(x1, y1, label="")

    x2, y2 = get_sample_points_isotropic_gaussian([0.0, -λ], σ₁, Int64(round(scale*weight[2])), δr, 
        num_round=num_round, to_shift=to_shift)
    scatter!(x2, y2, label="")
end

function polar_lines(radius, θ, δr; shift=[0.0, 0.0])
    r_pts = 0:δr:radius
    x = [r*cos(θ) for r in r_pts]
    y = [r*sin(θ) for r in r_pts]
    return x.+shift[1], y.+shift[2]
end

# scatter(get_sample_points(3, 8), xlims=(-4,4), ylims=(-4,4), size=(400,400), label="")

##################################
# Example 1
##################################
scale = 200; λ = 0.0
σ₀ = 1.0; σ₁ = 0.5
δr = 0.05; num_round = 4

xmin = -2.0; xmax = 2.0
ymin = -2.0; ymax = 2.0

fig1 = generate_samples_gaussian(scale, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax)

num_lines = 16
Δθ = 2*π/num_lines
for i = 1:num_lines
    x, y = polar_lines(3, i*Δθ, 0.01)
    plot!(x, y, color=:black, linewidth=1.5, label="")
end

savefig(fig1, folder*"geometry_rescaling.pdf")

##################################
# Example 2
##################################

scale = 200; λ = 2.0
σ₀ = 1.0; σ₁ = 1.0
δr = 0.1; num_round = 4

xmin = -2.5; xmax = 4.5
ymin = -2; ymax = 2

fig2 = generate_samples_gaussian(scale, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax)

num_lines = 16
Δy = (ymax-ymin)/(num_lines)

for i = 1:(num_lines-1)
    x = range(xmin, stop=xmax, length=100)
    y = (ymin + i*Δy)*ones(100)
    plot!(x, y, color=:black, linewidth=1.5, label="")
end

savefig(fig2, folder*"geometry_translation.pdf")

##################################
# Example 3
##################################

scale = 200; λ = 2.0
σ₀ = 1.0; σ₁ = 0.5
δr = 0.1; num_round = 4

xmin = -2.5; xmax = 4.5
ymin = -2; ymax = 2

fig3 = generate_samples_gaussian(scale, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax)

shift = [λ, 0]/(1-σ₁)
num_lines = 16
θinit = 3*π/4
θend = 2*π - θinit
Δθ = (θend - θinit)/(num_lines-1)
for i = 1:num_lines
    x, y = polar_lines(10, θinit + (i-1)*Δθ, 0.01, shift=shift)
    plot!(x, y, color=:black, linewidth=1.5, label="")
end

savefig(fig3, folder*"geometry_linear.pdf")

##################################
# Example 4
##################################

scale = 200; λ = 2.0
σ₀ = 1.0; σ₁ = 0.5
δr = 0.1; num_round = 4

xmin = -2.5; xmax = 4.5
ymin = -2; ymax = 2

fig4 = generate_samples_gaussian(scale, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax)

shift = [λ, 0] #/(1-σ₁)
num_lines = 16
θinit = 0.0
θend = 2*π - θinit
Δθ = (θend - θinit)/(num_lines-1)
for i = 1:num_lines
   x, y = polar_lines(10, θinit + (i-1)*Δθ, 0.01, shift=shift)
    plot!(x, y, color=:black, linewidth=1.5, label="")
end

savefig(fig4, folder*"geometry_linear_incorrect.pdf")

##################################
# Example 5
##################################

scale = 200; λ = 2.0
σ₀ = 1.0; σ₁ = 0.5
δr = 0.1; num_round = 4

xmin = -2.5; xmax = 4.5
ymin = -2; ymax = 2

fig5 = generate_samples_gaussian(scale, λ, σ₀, σ₁, δr, num_round, xmin, xmax, ymin, ymax)

shift = [λ/(1-σ₁^2), 0]
num_lines = 16
θinit = 0.0
θend = 2*π - θinit
Δθ = (θend - θinit)/(num_lines-1)
for i = 1:num_lines
   x, y = polar_lines(10, θinit + (i-1)*Δθ, 0.01, shift=shift)
    plot!(x, y, color=:black, linewidth=1.5, label="")
end

savefig(fig5, folder*"geometry_linear_incorrect_2.pdf")
