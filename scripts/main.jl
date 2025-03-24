using CUDA
using FacilityLocation
using KernelAbstractions
using BenchmarkTools

cpu = CPU()
gpu = CUDA.CUDABackend()

cpu_problem = FacilityLocationProblem(20, 500, 1000; backend=cpu, distance_cost=0.1)
gpu_problem = FacilityLocationProblem(20, 500, 1000; backend=gpu, distance_cost=0.1)

gpu_local_search(gpu_problem; iterations=100, verbose=true);

@btime gpu_local_search(gpu_problem; iterations=100);
@btime gpu_local_search(cpu_problem; iterations=100);
@btime local_search(cpu_problem; iterations=100);
