using Pkg
# Pkg.activate(joinpath(@__DIR__, "cuda"))
Pkg.activate(joinpath(@__DIR__, "metal"))

using Adapt
using Atomix: @atomic
using BenchmarkTools
using FacilityLocation
using KernelAbstractions
using Metal
using StableRNGs

backend = CPU()

distance_cost = 0.1
I, J, K = 3, 5, 20
problem = FacilityLocationProblem(StableRNG(0), Float32, I, J, K; backend, distance_cost)

"""
    local_search!(
        open_facilities,
        neighbor_costs_by_customer,
        neighbor_costs_total,
        setup_costs,
        serving_costs,
    )

# Arguments

- `open_facilities[i, k]`
- `neighbor_costs_by_customer[n, j, k]`
- `neighbor_costs_total[n, k]`

# Variables defined inside

- `neighbor_open_facilities[i, n, k]` contains `open_facilities[i, k]` except when `n = i`, in which case the value is switched
"""
@kernel function local_search3!(
    open_facilities,
    neighbor_costs_by_customer,
    neighbor_costs_total,
    setup_costs,
    serving_costs,
)
    @uniform I, J, K = size(serving_costs)
    i, j, _ = @index(Local, NTuple)  # i is the switching facility, j the affected customer
    n, _, _ = @index(Local, NTuple)  # i is the switching facility, j the affected customer
    _, _, k = @index(Group, NTuple)  # k is the instance
    neighbor_open_facilities = @localmem Bool (I, I)
    # initialize neighbors
    @assert I <= J
    if j <= I  # pretend j denotes a neighbor
        n2 = j
        o = open_facilities[i, k]
        neighbor_open_facilities[i, n2] = ifelse(n2 == i, !o, o)
    end
    @synchronize()
    # perform matmul to get customer costs in each neighbor
    tmp = typemax(eltype(serving_costs))
    for i2 in 1:I
        s = serving_costs[i2, j, k]
        tmp = ifelse(neighbor_open_facilities[i2, n], min(tmp, s), tmp)
    end
    neighbor_costs_by_customer[n, j, k] = tmp
    @synchronize()
    if j <= I  # pretend j denotes a facility
        i2 = j
        o = neighbor_open_facilities[i2, n]
        @atomic neighbor_costs_total[n, k] += o * setup_costs[i2, k]
    end
    if j == 1
        @atomic neighbor_costs_total[n, k] += neighbor_costs_by_customer[n, j, k]
    end
    @synchronize()
end

open_facilities = adapt(backend, ones(Bool, I, K));
neighbor_costs_by_customer = adapt(backend, zeros(Float32, I, J, K))
neighbor_costs_total = adapt(backend, zeros(Float32, I, K));
block_dims = (I, J, 1)
grid_dims = (I, J, K)
local_search3!(backend, block_dims)(
    open_facilities,
    neighbor_costs_by_customer,
    neighbor_costs_total,
    problem.setup_costs,
    problem.serving_costs;
    ndrange=grid_dims,
)
KernelAbstractions.synchronize(backend)

open_facilities
neighbor_costs_by_customer
