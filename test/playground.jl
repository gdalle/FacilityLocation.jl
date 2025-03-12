using Chairmarks
using LinearAlgebra, Statistics
using FacilityLocation
using KernelAbstractions, Metal
using Test

## Data

I, J, K = 20, 100, 10000
customers_per_facility = 5

facility_costs = rand(Int32(1):Int32(10), I, K);
customer_costs = rand(Int32(1):Int32(3), I, J, K);
open_facilities = rand(Bool, I, K);

backend = MetalBackend()
gpu_facility_costs = gpu(backend, facility_costs);
gpu_customer_costs = gpu(backend, customer_costs);
gpu_open_facilities = gpu(backend, open_facilities);

problem = MultipleFacilityLocationProblem(facility_costs, customer_costs)
gpu_problem = MultipleFacilityLocationProblem(gpu_facility_costs, gpu_customer_costs)

## Cost

@test total_cost(open_facilities, problem) ==
    gpu_total_cost(gpu_open_facilities, gpu_problem)

@be total_cost($open_facilities, $problem) seconds = 0.2
gpu_customer_choice_costs = allocate(backend, eltype(gpu_problem), J, K);
@be gpu_total_cost!($gpu_customer_choice_costs, $gpu_open_facilities, $gpu_problem) seconds =
    0.2
