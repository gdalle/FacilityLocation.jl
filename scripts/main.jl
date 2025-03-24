using Pkg
# Pkg.activate(joinpath(@__DIR__, "cuda"))
Pkg.activate(joinpath(@__DIR__, "metal"))

# using CUDA
using Metal

using FacilityLocation
using KernelAbstractions
using StableRNGs
using BenchmarkTools

cpu = CPU()
# gpu = CUDA.CUDABackend()
gpu = Metal.MetalBackend()

distance_cost = 0.1
I, J, K = 20, 500, 400
cpu_problem = FacilityLocationProblem(
    StableRNG(0), Float32, I, J, K; backend=cpu, distance_cost
)
gpu_problem = FacilityLocationProblem(
    StableRNG(0), Float32, I, J, K; backend=gpu, distance_cost
)

gpu_local_search(gpu_problem; iterations=100, verbose=true);

@btime gpu_local_search($gpu_problem; iterations=100);
@btime gpu_local_search($cpu_problem; iterations=100);
@btime local_search($cpu_problem; iterations=100);
