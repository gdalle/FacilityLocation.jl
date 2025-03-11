using FacilityLocation, FacilityLocationProblems
using KernelAbstractions, Metal

backend = MetalBackend()

problem = loadFacilityLocationProblem(:capa, typemax(Int))
gpu_problem = GPUFacilityLocationProblem(problem, backend)

solution = ones(Bool, length(problem.fixed_costs))
gpu_solution = KernelAbstractions.zeros(backend, Bool, length(problem.fixed_costs))
gpu_solution .= true

total_cost(solution, problem)
@profview total_cost(gpu_solution, gpu_problem)
