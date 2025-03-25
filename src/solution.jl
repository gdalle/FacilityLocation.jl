"""
    Solution

# Fields

- `open_facilities`: matrix such that `open_facilities[i, k]` is `true` if facility `i` is open in instance `k`
- `customer_assignments`: matrix such that `customer_assingments[j, k] = i` if customer `i` is assigned to facility `i` in instance `k`
"""
struct Solution{M1<:AbstractMatrix{Bool},M2<:AbstractMatrix{<:Integer}}
    open_facilities::M1
    customer_assignments::M2
end

"""
    Solution(open_facilities::AbstractMatrix, problem::FacilityLocationProblem)
"""
function Solution(open_facilities::AbstractMatrix{Bool}, problem::FLP)
    customer_assignments = similar(
        open_facilities, Int, nb_customers(problem), nb_instances(problem)
    )
    assign_customers!(customer_assignments, open_facilities, problem)
    return Solution(open_facilities, customer_assignments)
end

function Solution(problem)
    open_facilities = ones(Bool, nb_facilities(problem), nb_instances(problem))
    return Solution(open_facilities, problem)
end

function Base.copy(solution::Solution)
    return Solution(copy(solution.open_facilities), copy(solution.customer_assignments))
end
