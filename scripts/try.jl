using Pkg
# Pkg.activate(joinpath(@__DIR__, "cuda"))
Pkg.activate(joinpath(@__DIR__, "metal"))

using Adapt
using BenchmarkTools
using FacilityLocation
using KernelAbstractions
using Metal
using StableRNGs

backend = Metal.MetalBackend()

distance_cost = 0.3
I, J, K = 3, 10, 20

gpu_problem = FacilityLocationProblem(StableRNG(0), Float32, I, J, K; backend, distance_cost);
cpu_problem = FacilityLocationProblem(
    StableRNG(0), Float32, I, J, K; backend=CPU(), distance_cost
);

local_search(cpu_problem)[1].open_facilities
gpu_local_search(gpu_problem)
