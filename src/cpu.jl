"""
    Solution

# Constructors

    Solution(open_facilities::AbstractMatrix, problem)
    Solution(open_facilities::AbstractVector, problem)  # single-instance

# Fields

- `open_facilities`: matrix such that `open_facilities[i, k]` is `true` if facility `i` is open in instance `k`
- `customer_assignments`: matrix such that `customer_assingments[j, k] = i` if customer `i` is assigned to facility `i` in instance `k`
"""
struct Solution{M1<:AbstractMatrix{Bool},M2<:AbstractMatrix{<:Integer}}
    open_facilities::M1
    customer_assignments::M2
end

function Solution(open_facilities::AbstractMatrix{Bool}, problem::FLP)
    customer_assignments = similar(
        open_facilities, Int, nb_customers(problem), nb_instances(problem)
    )
    assign_customers!(customer_assignments, open_facilities, problem)
    return Solution(open_facilities, customer_assignments)
end

function Solution(open_facilities::AbstractVector{Bool}, problem::FLP)
    return Solution(reshape(open_facilities, length(open_facilities), 1), problem)
end

function Solution(open_facilities, problem::FLPWC)
    return Solution(open_facilities, problem.problem)
end

function plot_solution(solution::Solution, problem::FLPWC, k::Integer=1; kwargs...)
    fig = plot_instance(problem, k; kwargs...)
    open_facilities = solution.open_facilities[:, k]
    customer_assignments = solution.customer_assignments[:, k]

    facility_x = problem.facility_coordinates[:, 1, k]
    facility_y = problem.facility_coordinates[:, 2, k]

    scatter!(
        fig,
        facility_x[open_facilities],
        facility_y[open_facilities];
        markershape=:square,
        markercolor=:red,
        markersize=6,
        markerstrokecolor=:red,
        markerstrokewidth=2,
        label=nothing,
    )

    for j in customers(problem)
        i = customer_assignments[j]
        plot!(
            fig,
            [problem.facility_coordinates[i, 1, k], problem.customer_coordinates[j, 1, k]],
            [problem.facility_coordinates[i, 2, k], problem.customer_coordinates[j, 2, k]];
            color=:black,
            label=nothing,
        )
    end

    return fig
end

function Base.copy(solution::Solution)
    return Solution(copy(solution.open_facilities), copy(solution.customer_assignments))
end

function assign_customers!(
    customer_assignments::AbstractMatrix{Int},
    open_facilities::AbstractMatrix{Bool},
    problem::FLP,
)
    (; rank_to_facility) = problem
    @threads for k in instances(problem)
        for j in customers(problem)
            customer_assignments[j, k] = 0
            # find the best-ranking open facility
            for r in 1:nb_facilities(problem)
                i = rank_to_facility[r, j, k]
                if open_facilities[i, k]
                    customer_assignments[j, k] = i
                    break
                end
            end
            @assert customer_assignments[j, k] > 0
        end
    end
    return customer_assignments
end

function total_cost(solution::Solution, problem::FLP)
    (; setup_costs, serving_costs) = problem
    (; open_facilities, customer_assignments) = solution
    @assert size(open_facilities) == size(setup_costs)
    c = tmapreduce(+, instances(problem)) do k
        res = zero(eltype(problem))
        for i in facilities(problem)
            res += open_facilities[i, k] * setup_costs[i, k]
        end
        for j in customers(problem)
            i = customer_assignments[j, k]
            res += serving_costs[i, j, k]
        end
        return res
    end
    return c
end

function total_cost(solution::Solution, problem::FLPWC)
    return total_cost(solution, problem.problem)
end

function evaluate_flip!(flip_costs::AbstractMatrix, solution::Solution, problem::FLP)
    (; setup_costs, serving_costs, facility_to_rank, rank_to_facility) = problem
    (; open_facilities, customer_assignments) = solution
    @threads for k in instances(problem)
        for i in facilities(problem)
            if open_facilities[i, k]
                # closing saves on setup
                flip_costs[i, k] = -setup_costs[i, k]
            else
                # opening costs on setup
                flip_costs[i, k] = setup_costs[i, k]
            end
        end
        # more logical to iterate over customers than facilities for updating serving costs
        for j in customers(problem)
            # a customer switches if a better facility opens or if the current one closes
            i⁻ = customer_assignments[j, k]
            r⁻ = facility_to_rank[i⁻, j, k]
            # if better facility opens, switch to it directly
            for r⁺ in 1:(r⁻ - 1)
                i⁺ = rank_to_facility[r⁺, j, k]
                if !open_facilities[i⁺, k]
                    # opening saves on serving
                    cost_diff = serving_costs[i⁺, j, k] - serving_costs[i⁻, j, k]
                    @assert cost_diff <= 0
                    flip_costs[i⁺, k] += cost_diff
                end
            end
            # if current facility closes, find the next highest-ranking open facility
            found = false
            for r⁺ in (r⁻ + 1):nb_facilities(problem)
                i⁺ = rank_to_facility[r⁺, j, k]
                if open_facilities[i⁺, k]
                    found = true
                    # closing costs on serving
                    cost_diff = serving_costs[i⁺, j, k] - serving_costs[i⁻, j, k]
                    @assert cost_diff >= 0
                    flip_costs[i⁻, k] += cost_diff
                    break
                end
            end
            if !found
                # switch would leave a customer stranded, infeasible
                flip_costs[i⁻, k] = typemax(eltype(problem))
            end
        end
    end
    return flip_costs
end

function perform_best_flip!(flip_costs::AbstractMatrix, solution::Solution, problem::FLP)
    (; open_facilities, customer_assignments) = solution
    flipped = false
    @threads for k in instances(problem)
        i = argmin(view(flip_costs, :, k))
        if flip_costs[i, k] < 0
            flipped = true
            open_facilities[i, k] = !open_facilities[i, k]
        end
    end
    assign_customers!(customer_assignments, open_facilities, problem)
    return flipped
end

"""
    local_search(old_solution, problem; iterations=10)

Perform local search by iteratively finding and applying the best flip movement (open or close a single facility) across all instances in parallel.

Return a tuple `(new_solution, cost_evolution)`.
"""
function local_search(old_solution::Solution, problem::FLP; iterations=10)
    solution = copy(old_solution)
    flip_costs = fill(
        typemax(eltype(problem)), nb_facilities(problem), nb_instances(problem)
    )
    cost_evolution = fill(convert(eltype(problem), NaN), iterations + 1)
    cost_evolution[1] = total_cost(solution, problem)
    for it in 1:iterations
        evaluate_flip!(flip_costs, solution, problem)
        flipped = perform_best_flip!(flip_costs, solution, problem)
        cost_evolution[it + 1] = total_cost(solution, problem)
        if !flipped
            resize!(cost_evolution, it)
            break
        end
    end
    return solution, cost_evolution
end

function local_search(
    problem::FLPWC;
    iterations=10,
    starting_solution=Solution(
        ones(Bool, nb_facilities(problem), nb_instances(problem)), problem
    ),
)
    return local_search(starting_solution, problem.problem; iterations=iterations)
end
