# This script intends to install all Julia pckages needed
packages = ["BasicInterpolators",
    "BenchmarkTools",
    "Conda",
    "CUDA",
    "CSV",
    "DataFrames",
    "Documenter",
    "FFTW",
    "FileIO",
    "Flux",
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
    Pkg.add(packagename)
end
