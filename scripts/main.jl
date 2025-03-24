using CUDA
using FacilityLocation
using KernelAbstractions
using BenchmarkTools

cpu = CPU()
gpu = CUDA.CUDABackend()

distance_cost = 0.1
I, J, K = 20, 500, 400
cpu_problem = FacilityLocationProblem(I, J, K; backend=cpu, distance_cost)
gpu_problem = FacilityLocationProblem(I, J, K; backend=gpu, distance_cost)

gpu_local_search(gpu_problem; iterations=100, verbose=true);

@btime gpu_local_search(gpu_problem; iterations=100);
@btime gpu_local_search(cpu_problem; iterations=100);
@btime local_search(cpu_problem; iterations=100);
