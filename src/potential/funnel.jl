export generate_Funnel_Potential

function generate_Funnel_Potential(n, σf; cutoff=-200)
    function U(x)
        x₁ = x[1]
        return x₁^2/(2*σf^2) + x₁*(n-1)/2 + exp(-max(x₁,cutoff))/2*(dot(x,x)-x₁^2)
    end
    
    function gradU(x)
        x₁ = x[1]
        v = zeros(n)
        v[1] = x₁/σf^2 + (n-1)/2 - exp(-max(x₁,cutoff))/2*(dot(x,x)-x₁^2)
        v[2:end] = exp(-max(x₁,cutoff)).*x[2:end]
        return v
    end
    
    function laplaceU(x)
        x₁ = x[1]
        return exp(-x₁)*(n-1) + 1/σf^2 + 1/2*exp(-x₁)*dot(x[2:end],x[2:end])
    end

    function hessU(x)
        x₁ = x[1]
        row1 = vcat(1/σf^2 + 1/2*exp(-x₁)*dot(x[2:end],x[2:end]), -exp(-x₁)*x[2:end])
        row2 = hcat(-exp(-x₁)*x[2:end], exp(-x₁)*Matrix(1.0I,n-1,n-1))
        return vcat(row1', row2)
    end
    
    return FixedPotential(n, U, gradU, hessU, laplaceU, [σf], nothing)
end
