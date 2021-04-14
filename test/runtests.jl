using ResNet9
using Test
using Flux
using CUDA

if CUDA.has_cuda_gpu()
    #CUDA.allowscalar(false)
    device = gpu
else
    device = cpu
end

@testset "ResNet9.jl" begin
    @testset "Two classes" begin
        model = resnet9(ichs = 3, ncls = 2) |> device
        x = Float32.(rand(32, 32, 3, 4)) |> device
        ŷ = model(x)
        @test size(ŷ) == (1, 4)
    end

    @testset "Five classes" begin
        model = resnet9(ichs = 3, ncls = 5) |> device
        x = Float32.(rand(32, 32, 3, 4)) |> device
        ŷ = model(x)
        @test size(ŷ) == (5, 4)
    end
end