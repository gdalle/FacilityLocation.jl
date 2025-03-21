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

"""
    FacilityLocationProblemWithCoordinates

Version of `FacilityLocationProblem` with coordinates for facilities and customers.
Useful for plotting.

# Fields
- `problem`: a `FacilityLocationProblem`
- `facility_coordinates`: a 3d-array such that `facility_coordinates[i, :, k]` is the 2d-coordinates of facility `i` in instance `k`
- `customer_coordinates`: a 3d-array such that `customer_coordinates[j, :, k]` is the 2d-coordinates of customer `j` in instance `k`
"""
struct FacilityLocationProblemWithCoordinates{P<:FLP,A3i<:AbstractArray{<:Real,3}}
    problem::P
    facility_coordinates::A3i
    customer_coordinates::A3i
end

const FLPWC = FacilityLocationProblemWithCoordinates

Base.eltype(::FLPWC{P,A3i}) where {P,A3i} = eltype(P)

nb_instances(problem::FLPWC) = nb_instances(problem.problem)
nb_facilities(problem::FLPWC) = nb_facilities(problem.problem)
nb_customers(problem::FLPWC) = nb_customers(problem.problem)

instances(problem::FLPWC) = instances(problem.problem)
facilities(problem::FLPWC) = facilities(problem.problem)
customers(problem::FLPWC) = customers(problem.problem)

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
    return FacilityLocationProblemWithCoordinates(
        FLP(setup_costs, serving_costs), facility_coordinates, customer_coordinates
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

    serving_costs = zeros(I, J)
    for j in 1:J, i in 1:I
        dx = facility_coordinates[i, 1] - customer_coordinates[j, 1]
        dy = facility_coordinates[i, 2] - customer_coordinates[j, 2]
        serving_costs[i, j] = sqrt(dx * dx + dy * dy)
    end
    return FacilityLocationProblemWithCoordinates(
        FLP(setup_costs, serving_costs),
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

function plot_instance(problem::FLPWC, k::Integer=1; kwargs...)
    @assert k in instances(problem)

    facility_x = problem.facility_coordinates[:, 1, k]
    facility_y = problem.facility_coordinates[:, 2, k]
    customer_x = problem.customer_coordinates[:, 1, k]
    customer_y = problem.customer_coordinates[:, 2, k]

    fig = plot(; xrange=(0, 1), yrange=(0, 1), kwargs...)
    scatter!(
        fig,
        facility_x,
        facility_y;
        label="Facilities",
        markersize=6,
        marker=:square,
        markercolor=:white,
        markerstrokewidth=2,
        markerstrokecolor=:red,
    )
    scatter!(
        fig,
        customer_x,
        customer_y;
        label="Customers",
        markersize=3,
        marker=:circle,
        markercolor=:blue,
    )
    return fig
end
