using ModGrid
using CSV
using DataFrames
using Test

export load_log_gaussian_cox_process

function load_log_gaussian_cox_process(; M = 30, σ₁ = sqrt(1.91), β = 1/33, folder="./", runtest=false)

    μ₁ = log(126) - σ₁^2/2
    x_lb = -5
    x_rb = 5
    y_lb = -8
    y_rb = 2

    df = DataFrame(CSV.File(folder*"src/data/finpines.csv"))
    locx = df.x
    locy = df.y;
    num_data = length(locx)

    locx_rescale = (locx.-x_lb)/(x_rb-x_lb)
    locy_rescale = (locy.-y_lb)/(y_rb-y_lb)

    Δx = 1/M; Δy = 1/M
    a = Δx*Δy
    Ytable = zeros(M, M)
    for i = 1:num_data
        idx_x = Int64(ceil(locx_rescale[i]/Δx))
        idx_x == 0 ? idx_x = 1 : nothing
        idx_y = Int64(ceil(locy_rescale[i]/Δy))
        idx_y == 0 ? idx_y = 1 : nothing
        Ytable[idx_x, idx_y] += 1
    end


    Σ₁ = zeros(M^2, M^2)
    for idx1 = 1:M^2
        for idx2 = 1:M^2
            dist = shortest_distance_2d_grid(rescaleidx(idx1, M), rescaleidx(idx2, M), M)
            Σ₁[idx1, idx2] = σ₁^2*exp(-dist/(30*β))
        end
    end

    n = M^2
    U₀ = generate_Gaussian_Potential(n, zeros(n), 1.0);

    C = cholesky(Σ₁)
    L = C.L;
    U = C.U;

    y = reshape(Ytable, n)
    yTL = y'*L
    Uy = U*y
    Usq = U.^2

    diffU(x) = -μ₁*sum(y) - yTL*x + a*exp(μ₁)*sum(exp.(L*x))
    diffU_grad(x) = -Uy + a*exp(μ₁)*U*exp.(L*x)
    diffU_hess(x) =  a*exp(μ₁)*U*Diagonal(exp.(L*x))*L
    diffU_lap(x) = a*exp(μ₁)*sum(Usq*exp.(L*x))

    U₁ = FixedPotential(n, x->U₀.U(x)+diffU(x),
        x->U₀.gradU(x)+diffU_grad(x),
        x->U₀.HessU(x)+diffU_hess(x),
        x->U₀.LaplaceU(x)+diffU_lap(x),
        [], nothing);

	if runtest   
		@testset "test U₁" begin
			n = M^2
			for trial = 1:100
				x = randn(n)
				δx = randn(n); δx = δx/norm(δx);
				ϵ = 1.0e-3
				@test abs((U₁.U(x+ϵ*δx) - U₁.U(x-ϵ*δx))/(2*ϵ) - dot(U₁.gradU(x), δx)) < ϵ
				@test norm((U₁.gradU(x+ϵ*δx) - U₁.gradU(x-ϵ*δx))/(2*ϵ) - U₁.HessU(x)*δx) < ϵ
				@test abs(tr(U₁.HessU(x))/U₁.LaplaceU(x) - 1) < ϵ
			end
		end 
	end

    return U₀, U₁, locx_rescale, locy_rescale, Ytable, Σ₁ 
end
