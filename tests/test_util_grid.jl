push!(LOAD_PATH,"../src")
using Test
using NEIS

@testset "test rescaleidx" begin
    M = 30
    for k = 1:1000
        i = Int64(ceil(rand()*M))
        j = Int64(ceil(rand()*M))
        newi, newj = rescaleidx(rescaleidx((i,j), M), M)
        @test newi == i
        @test newj == j
    end
    @test shortest_distance_2d_grid((1,1), (M,M), M) == sqrt(2)
    @test shortest_distance_2d_grid((1,1), (M-1,M), M) == sqrt(5)
end
