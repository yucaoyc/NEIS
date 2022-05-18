#!/bin/bash
echo "running geometry"
julia geometry.jl

echo "running generator-1d-gaussian"
julia generator-1d-gaussian.jl

echo "running generator-poisson-neumann"
julia generator-poisson-neumann.jl

echo "running poisson-torus"
julia poisson_torus.jl

echo "running poisson-neumann"
julia poisson_neumann.jl
