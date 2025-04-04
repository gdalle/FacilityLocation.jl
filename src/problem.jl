"""
    FacilityLocationProblem

# Fields

- `serving_costs`: a 3d-array such that `serving_costs[i, j, k]` is the cost of serving customer `j` with facility `i` in instance `k`
- `setup_costs`: a matrix such that `setup_costs[i, k]` is the cost of opening facility `i` in instance `k`
- `rank_to_facility`: a 3d-array such that `rank_to_facility[r, j, k]` is the integer index of the `r`-th closest facility to customer `j` in instance `k`
- `facility_to_rank`: a 3d-array such that `facility_to_rank[i, j, k]` is the rank of facility `i` for customer `j` in instance `k`
- `facility_coordinates`: either `nothing` or a matrix of tuples such that `(x, y) = facility_coordinates[i, k]` are the latitude and longitude of facility `i` in instance `k`
- `customer_coordinates`: either `nothing` or a matrix of tuples such that `(x, y) = customer_coordinates[j, k]` are the latitude and longitude of customer `j` in instance `k`
"""
struct FacilityLocationProblem{
    Ti<:Integer,
    Tr<:Real,
    A2r<:AbstractArray{Tr,2},
    A3r<:AbstractArray{Tr,3},
    A3i<:AbstractArray{Ti,3},
    C<:Union{AbstractMatrix{Tuple{Tr,Tr}},Nothing},
}
    setup_costs::A2r
    serving_costs::A3r
    rank_to_facility::A3i
    facility_to_rank::A3i
    facility_coordinates::C
    customer_coordinates::C
end

"""
    FacilityLocationProblem(
        setup_costs::AbstractMatrix,
        serving_costs::AbstractArray{_,3};
        facility_coordinates=nothing,
        customer_coordinates=nothing,
    )
"""
function FacilityLocationProblem(
    setup_costs::AbstractMatrix,
    serving_costs::AbstractArray{<:Real,3};
    facility_coordinates=nothing,
    customer_coordinates=nothing,
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
        setup_costs,
        serving_costs,
        rank_to_facility,
        facility_to_rank,
        facility_coordinates,
        customer_coordinates,
    )
end

"""
    FacilityLocationProblem(
        setup_costs::AbstractMatrix,
        facility_coordinates::AbstractMatrix,
        customer_coordinates::AbstractMatrix;
        distance_cost=1
    )
"""
function FacilityLocationProblem(
    setup_costs::AbstractMatrix,
    facility_coordinates::AbstractMatrix{<:Tuple{Real,Real}},
    customer_coordinates::AbstractMatrix{<:Tuple{Real,Real}};
    distance_cost=one(eltype(setup_costs)),
)
    I, K = size(setup_costs)
    J, K2 = size(customer_coordinates)
    I2, K3 = size(facility_coordinates)
    @assert I == I2
    @assert K == K2 == K3
    serving_costs = zeros(Base.promote_eltype(setup_costs, distance_cost), I, J, K)
    for k in 1:K, j in 1:J, i in 1:I
        coord_diff = facility_coordinates[i, k] .- customer_coordinates[j, k]
        serving_costs[i, j, k] = distance_cost * sqrt(sum(abs2, coord_diff))
    end
    return FacilityLocationProblem(
        setup_costs, serving_costs; facility_coordinates, customer_coordinates
    )
end

"""
    FacilityLocationProblem(
        I, J, K;
        distance_cost=1,
        seed, rng,
    )
"""
function FacilityLocationProblem(
    I::Integer, J::Integer, K::Integer=1; distance_cost=1, seed=0, rng=StableRNG(seed)
)
    setup_costs = rand(rng, Float32, I, K)
    facility_coordinates = [(rand(rng, Float32), rand(rng, Float32)) for i in 1:I, k in 1:K]
    customer_coordinates = [(rand(rng, Float32), rand(rng, Float32)) for j in 1:J, k in 1:K]
    # not obvious to parametrize with `customers_per_facility` here because we don't know the average distance between a customer and its closest neighboring facility
    return FacilityLocationProblem(
        setup_costs, facility_coordinates, customer_coordinates; distance_cost
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
