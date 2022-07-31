# This script intends to install all Julia pckages needed
packages = ["BasicInterpolators",
    "BenchmarkTools",
    "Conda",
    "CUDA",
    "CSV",
    "DataFrames",
    "Documenter",
    "Distributions",
    "FFTW",
    "FileIO",
    ["Flux","0.12.9"],
    "ForwardDiff",
    "GR",
    "JLD2",
    "LaTeXStrings",
    "Plots",
    "PyPlot",
    "SpecialFunctions",
    "StatsPlots",
    "Statistics"
    ]

using Pkg

for packagename in packages
    if typeof(packagename) <: AbstractString
        Pkg.add(packagename)
    else
        Pkg.add(name=packagename[1], version=packagename[2])
    end
end
