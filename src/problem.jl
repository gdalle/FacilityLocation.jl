"""
    FacilityLocationProblem

# Constructors

    FacilityLocationProblem(setup_costs::AbstractMatrix, serving_costs::AbstractArray{<:Real,3})
    FacilityLocationProblem(setup_costs::AbstractVector, serving_costs::AbstractMatrix)  # single-instance

# Fields

- `setup_costs`: a matrix such that `setup_costs[i, k]` is the cost of opening facility `i` in instance `k`
- `serving_costs`: a 3d-array such that `serving_costs[i, j, k]` is the cost of serving customer `j` with facility `i` in instance `k`
- `rank_to_facility`: a 3d-array such that `rank_to_facility[r, j, k]` is the integer index of the `r`-th closest facility to customer `j` in instance `k`
- `facility_to_rank`: a 3d-array such that `facility_to_rank[i, j, k]` is the rank of facility `i` for customer `j` in instance `k`
"""
struct FacilityLocationProblem{
    Ti<:Integer,
    Tr<:Real,
    A2r<:AbstractArray{Tr,2},
    A3r<:AbstractArray{Tr,3},
    A3i<:AbstractArray{Ti,3},
}
    setup_costs::A2r
    serving_costs::A3r
    rank_to_facility::A3i
    facility_to_rank::A3i
end

function FacilityLocationProblem(
    setup_costs::AbstractMatrix, serving_costs::AbstractArray{<:Real,3}
)
    @assert eltype(setup_costs) == eltype(serving_costs)
    I1, K1 = size(setup_costs)
    I2, J2, K2 = size(serving_costs)
    @assert I1 == I2
    @assert K1 == K2
    @assert get_backend(setup_costs) == get_backend(serving_costs)

    rank_to_facility = similar(serving_costs, Int)
    facility_to_rank = similar(serving_costs, Int)
    for k in 1:K2, j in 1:J2
        costs = view(serving_costs, :, j, k)
        facilities = sortperm(costs)
        ranks = invperm(facilities)
        rank_to_facility[:, j, k] .= facilities
        facility_to_rank[:, j, k] .= ranks
    end
    return FacilityLocationProblem(
        setup_costs, serving_costs, rank_to_facility, facility_to_rank
    )
end

function FacilityLocationProblem(setup_costs::AbstractVector, serving_costs::AbstractMatrix)
    I, J = size(serving_costs)
    return FacilityLocationProblem(
        reshape(setup_costs, I, 1), reshape(serving_costs, I, J, 1)
    )
end

const FLP = FacilityLocationProblem

Base.eltype(::FLP{Ti,Tr}) where {Ti,Tr} = Tr

nb_instances(problem::FLP) = size(problem.setup_costs, 2)
nb_facilities(problem::FLP) = size(problem.setup_costs, 1)
nb_customers(problem::FLP) = size(problem.serving_costs, 2)

instances(problem::FLP) = 1:nb_instances(problem)
facilities(problem::FLP) = 1:nb_facilities(problem)
customers(problem::FLP) = 1:nb_customers(problem)

function KernelAbstractions.get_backend(problem::FLP)
    return get_backend(problem.setup_costs)
end
