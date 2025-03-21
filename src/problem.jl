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
    C<:Union{AbstractArray{Tr,3},Nothing},
    A2r<:AbstractArray{Tr,2},
    A3r<:AbstractArray{Tr,3},
    A3i<:AbstractArray{Ti,3},
}
    setup_costs::A2r
    serving_costs::A3r
    rank_to_facility::A3i
    facility_to_rank::A3i
    facility_coordinates::C
    customer_coordinates::C
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
        setup_costs, serving_costs, rank_to_facility, facility_to_rank, nothing, nothing
    )
end

function FacilityLocationProblem(setup_costs::AbstractVector, serving_costs::AbstractMatrix)
    I, J = size(serving_costs)
    return FacilityLocationProblem(
        reshape(setup_costs, I, 1), reshape(serving_costs, I, J, 1)
    )
end

function FacilityLocationProblem(
    setup_costs::AbstractMatrix,
    facility_coordinates::AbstractArray{<:Real,3},
    customer_coordinates::AbstractArray{<:Real,3},
)
    I, K = size(setup_costs)
    J, D, K2 = size(customer_coordinates)
    I2, D2, K3 = size(facility_coordinates)
    @assert I == I2
    @assert K == K2 == K3
    @assert D == D2 == 2
    @assert eltype(setup_costs) ==
        eltype(facility_coordinates) ==
        eltype(customer_coordinates)

    serving_costs = zeros(eltype(setup_costs), I, J, K)
    for k in 1:K, j in 1:J, i in 1:I
        dx = facility_coordinates[i, 1, k] - customer_coordinates[j, 1, k]
        dy = facility_coordinates[i, 2, k] - customer_coordinates[j, 2, k]
        serving_costs[i, j, k] = sqrt(dx * dx + dy * dy)
    end

    p = FacilityLocationProblem(setup_costs, serving_costs)
    return FacilityLocationProblem(
        p.setup_costs,
        p.serving_costs,
        p.rank_to_facility,
        p.facility_to_rank,
        facility_coordinates,
        customer_coordinates,
    )
end

function FacilityLocationProblem(
    setup_costs::AbstractVector,
    facility_coordinates::AbstractMatrix,
    customer_coordinates::AbstractMatrix,
)
    I = length(setup_costs)
    J, D = size(customer_coordinates)
    I2, D2 = size(facility_coordinates)
    @assert I == I2
    @assert D == D2 == 2

    return FacilityLocationProblem(
        reshape(setup_costs, I, 1),
        reshape(facility_coordinates, I, D, 1),
        reshape(customer_coordinates, J, D, 1),
    )
end

function FacilityLocationProblem(
    I::Integer,
    J::Integer,
    K::Integer=1;
    customers_per_facility=10,
    seed=0,
    rng=StableRNG(seed),
)
    setup_costs = rand(rng, Float32, I, K) * customers_per_facility
    facility_coordinates = rand(rng, Float32, I, 2, K)
    customer_coordinates = rand(rng, Float32, J, 2, K)
    return FacilityLocationProblem(setup_costs, facility_coordinates, customer_coordinates)
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
