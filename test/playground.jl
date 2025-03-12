using Chairmarks
using LinearAlgebra, Statistics
using FacilityLocation
using Adapt, KernelAbstractions, Metal
using StableRNGs
using Test

## Data

I, J, K = 50, 100, 1000
customers_per_facility = 5

facility_costs = rand(StableRNG(0), Float32, I, K);
customer_costs = rand(StableRNG(1), Float32, I, J, K);
open_facilities = rand(StableRNG(2), Bool, I, K);

backend = MetalBackend()
gpu_facility_costs = adapt(backend, facility_costs);
gpu_customer_costs = adapt(backend, customer_costs);
gpu_open_facilities = adapt(backend, open_facilities);

problem = MultipleFacilityLocationProblem(facility_costs, customer_costs)
gpu_problem = MultipleFacilityLocationProblem(gpu_facility_costs, gpu_customer_costs)

## Cost

total_cost(open_facilities, problem)
gpu_total_cost(gpu_open_facilities, gpu_problem)

@test total_cost(open_facilities, problem) ≈
    gpu_total_cost(gpu_open_facilities, gpu_problem)

@be total_cost($open_facilities, $problem) seconds = 0.2
@be gpu_total_cost($gpu_open_facilities, $gpu_problem) seconds = 0.2

## Local search

open_facilities = zeros(Bool, I, K)
open_facilities[1, :] .= true

local_search(open_facilities, problem; iterations=100)
