using ResNet9
using Test
using Flux
using CUDA

if CUDA.has_cuda_gpu()
    CUDA.allowscalar(false)
    device = gpu
else
    device = cpu
end

@testset "ResNet9.jl" begin
    batchsize = 2
    inchs = 3
    x1 = Float32.(rand(16, 16, inchs, batchsize)) |> device
    x2 = Float32.(rand(16, 16, 2 * inchs, batchsize)) |> device

    @testset "Two classes" begin
        model = resnet9(inchannels = inchs, nclasses = 2) |> device
        ŷ = model(x1)
        @test size(ŷ) == (1, batchsize)
    end

    @testset "Five classes" begin
        model = resnet9(inchannels = inchs, nclasses = 5) |> device
        ŷ = model(x1)
        @test size(ŷ) == (5, batchsize)
    end

    @testset "Transfer learning" begin
        model = resnet9(inchannels = inchs, nclasses = 5)
        # Change the number of classes and add dropout
        model2 = resnet9(model, nclasses = 10, dropout = 0.5) |> device
        ŷ = model2(x1)
        @test size(ŷ) == (10, batchsize)
        # Change the number of input channels
        model2 = resnet9(model, inchannels = 2 * inchs) |> device
        ŷ = model2(x2)
        @test size(ŷ) == (10, batchsize)
    end
end