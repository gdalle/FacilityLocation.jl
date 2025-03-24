using CUDA
using FacilityLocation
using KernelAbstractions

using Plots

# backend = CPU()
backend = CUDA.CUDABackend()
problem = FacilityLocationProblem(20, 500, 2; backend, distance_cost=0.1)

solution, _ = local_search(p)

plot_solution(solution, p, 1)

adapt(backend, ones(10))
