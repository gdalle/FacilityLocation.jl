using LinearAlgebra, Statistics
using FacilityLocation
using StableRNGs
using Test

@testset "Small instance" begin
    customers_per_facility = 2
    setup_costs = [1.0, 2.0, 3.0]
    serving_costs = [
        0.1 0.2 0.5 0.7 0.4 0.7
        0.2 0.6 0.3 0.1 0.6 0.6
        0.3 0.4 0.7 0.4 0.2 0.5
    ]

    problem = FacilityLocationProblem(setup_costs, serving_costs)

    open_facilities = [false, true, true]
    solution = Solution(open_facilities, problem)
    new_solution, _ = local_search(problem, solution)

    @test nb_instances(problem) == 1
    @test nb_facilities(problem) == 3
    @test nb_customers(problem) == 6

    @test problem.rank_to_facility[:, :, 1] == [
        1 1 2 2 3 3
        2 3 1 3 1 2
        3 2 3 1 2 1
    ]
    @test problem.facility_to_rank[:, :, 1] == [
        1 1 2 3 2 3
        2 3 1 1 3 2
        3 2 3 2 1 1
    ]

    @test solution.customer_assignments[:, 1] == [2, 3, 2, 2, 3, 3]
    @test total_cost(solution, problem) == 2 + 3 + 0.2 + 0.4 + 0.3 + 0.1 + 0.2 + 0.5

    @test all(==(2), new_solution.customer_assignments)
    @test total_cost(new_solution, problem) == 2 + sum(serving_costs[2, :])
end

@testset "Large instance" begin
    I, J, K = 20, 500, 5
    customers_per_facility = 10
    setup_costs = rand(StableRNG(0), Float32, I, K) * customers_per_facility
    serving_costs = rand(StableRNG(1), Float32, I, J, K)
    problem = FacilityLocationProblem(setup_costs, serving_costs)

    solution = Solution(ones(Bool, I, K), problem)
    new_solution, cost_evolution = local_search(problem, solution; iterations=100)
    @test total_cost(new_solution, problem) < total_cost(solution, problem)
    @test length(cost_evolution) > 5
    @test all(>(0), cost_evolution)
    @test mean(new_solution.open_facilities) < 1
end

@testset "Instances with coordinates" begin
    I, J, K = 20, 500, 5
    problem = FacilityLocationProblem(I, J, K)
    solution = Solution(ones(Bool, I, K), problem)
    new_solution, cost_evolution = local_search(problem, solution; iterations=100)
    @test total_cost(new_solution, problem) < total_cost(solution, problem)
    @test length(cost_evolution) > 5
    @test all(>(0), cost_evolution)
    @test mean(new_solution.open_facilities) < 1
end
