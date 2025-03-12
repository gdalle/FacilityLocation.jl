"""
    MultipleFacilityLocationProblem

# Fields

- `facility_costs`: a matrix such that `facility_costs[i, k]` is the cost of opening facility `i` in instance `k`
- `customer_costs`: a 3d-array such that `customer_costs[i, j, k]` is the cost of serving customer `j` with facility `i` in instance `k`
"""
struct MultipleFacilityLocationProblem{
    T<:Real,A2<:AbstractArray{T,2},A3<:AbstractArray{T,3}
}
    facility_costs::A2
    customer_costs::A3

    function MultipleFacilityLocationProblem(facility_costs, customer_costs)
        @assert eltype(facility_costs) == eltype(customer_costs)
        I1, K1 = size(facility_costs)
        I2, J2, K2 = size(customer_costs)
        @assert I1 == I2
        @assert K1 == K2
        @assert get_backend(facility_costs) == get_backend(customer_costs)
        return new{eltype(facility_costs),typeof(facility_costs),typeof(customer_costs)}(
            facility_costs, customer_costs
        )
    end
end

const MFLP = MultipleFacilityLocationProblem

Base.eltype(::MFLP{T}) where {T} = T

nb_instances(problem::MFLP) = size(problem.facility_costs, 2)
nb_facilities(problem::MFLP) = size(problem.facility_costs, 1)
nb_customers(problem::MFLP) = size(problem.customer_costs, 2)

instances(problem::MFLP) = 1:nb_instances(problem)
facilities(problem::MFLP) = 1:nb_facilities(problem)
customers(problem::MFLP) = 1:nb_customers(problem)

function KernelAbstractions.get_backend(problem::MFLP)
    return get_backend(problem.facility_costs)
end
