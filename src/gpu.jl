@kernel function gpu_local_search!(
    open_facilities::AbstractMatrix,
    current_costs_by_customer::AbstractMatrix,
    neighbor_costs_by_customer::AbstractArray{<:Any,3},
    neighbor_switch_costs::AbstractMatrix,
    best_neighbor_cost::AbstractVector,
    best_neighbor_index::AbstractVector,
    setup_costs::AbstractMatrix,
    serving_costs::AbstractArray{<:Any,3},
    iterations::Integer,
    ::Val{I},
    ::Val{K},
) where {I,K}
    @uniform J = size(serving_costs, 2)
    @assert I <= J

    # i is the current facility, n the switching facility, j the affected customer
    i, j, _ = @index(Local, NTuple)
    n, _, _ = @index(Local, NTuple)
    # k is the instance
    _, _, k = @index(Group, NTuple)

    # initialize local memory
    neighbor_open_facilities = @localmem Bool (I, I)

    for _ in 1:iterations

        # initialize neighbors
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
        # perform matmul to get customer costs in the current solution
        if n == 1
            tmp = typemax(eltype(serving_costs))
            for i2 in 1:I
                s = serving_costs[i2, j, k]
                tmp = ifelse(open_facilities[i2, k], min(tmp, s), tmp)
            end
            current_costs_by_customer[j, k] = tmp
        end
        @synchronize()

        # compare customer and setup costs between neighbor and current
        if j <= I  # pretend j denotes a facility
            i2 = j
            o = neighbor_open_facilities[i2, n] - open_facilities[i2, k]
            @atomic neighbor_switch_costs[n, k] += o * setup_costs[i2, k]
        end
        @synchronize()

        @atomic neighbor_switch_costs[n, k] += (
            neighbor_costs_by_customer[n, j, k] - current_costs_by_customer[j, k]
        )
        @synchronize()

        # find best neighbor
        if j == 1 && n == 1
            # TODO: better way to do min inside a kernel
            best_neighbor_cost[k] = typemax(Float32)
            best_neighbor_index[k] = typemax(Int)
            for n2 in 1:I
                best_neighbor_cost[k] = min(
                    neighbor_switch_costs[n2, k], best_neighbor_cost[k]
                )
                best_neighbor_index[k] = ifelse(
                    best_neighbor_cost[k] == neighbor_switch_costs[n2, k],
                    n2,
                    best_neighbor_index[k],
                )
            end
        end
        @synchronize()
        if best_neighbor_cost[k] < 0 && j == 1
            o = open_facilities[n, k]
            open_facilities[n, k] = ifelse(n == best_neighbor_index[k], !o, o)
        end
        @synchronize()
    end
end

function gpu_local_search(problem::FLP; iterations=10)
    I, J, K = nb_facilities(problem), nb_customers(problem), nb_instances(problem)
    backend = get_backend(problem)

    open_facilities = adapt(backend, ones(Bool, I, K))
    current_costs_by_customer = adapt(backend, zeros(Float32, J, K))
    neighbor_costs_by_customer = adapt(backend, zeros(Float32, I, J, K))
    neighbor_switch_costs = adapt(backend, zeros(Float32, I, K))
    best_neighbor_cost = adapt(backend, zeros(Float32, K))
    best_neighbor_index = adapt(backend, zeros(Int, K))
    block_dims = (I, J, 1)
    grid_dims = (I, J, K)

    gpu_local_search!(backend, block_dims)(
        open_facilities,
        current_costs_by_customer,
        neighbor_costs_by_customer,
        neighbor_switch_costs,
        best_neighbor_cost,
        best_neighbor_index,
        problem.setup_costs,
        problem.serving_costs,
        iterations,
        Val(I),
        Val(K);
        ndrange=grid_dims,
    )
    KernelAbstractions.synchronize(backend)
    return open_facilities
end
