export cauchy, cauchyderi

function cauchy(x, μ, σ)
    """
        Cauchy distributions
    """
    1/(π*σ*(1+(x-μ)^2/σ^2))
end

function cauchyderi(x, μ, σ)
    """
        return ∇π where π follows Cauchy distributions
    """
    d = x-μ
    return -2*d*σ/π/(σ^2+d^2)^2
end
