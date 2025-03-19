@kernel function choose_facilities_coarse!(
    customer_choice_costs::AbstractVector{T},
    @Const(open_facilities::AbstractMatrix{Bool}),
    @Const(serving_costs::AbstractArray{T,3}),
) where {T}
    k = @index(Global)
    ck = zero(T)
    @simd for j in axes(serving_costs, 2)
        cjk = typemax(T)
        @simd for i in axes(open_facilities, 1)
            new_cjk = min(cjk, serving_costs[i, j, k])
            cjk = ifelse(open_facilities[i, k], new_cjk, cjk)
        end
        ck += cjk
    end
    customer_choice_costs[k] = ck
end

@kernel function choose_facilities!(
    customer_choice_costs::AbstractMatrix{T},
    @Const(open_facilities::AbstractMatrix{Bool}),
    @Const(serving_costs::AbstractArray{T,3}),
) where {T}
    j, k = @index(Global, NTuple)
    cjk = typemax(T)
    @simd for i in axes(open_facilities, 1)
        new_cjk = min(cjk, serving_costs[i, j, k])
        cjk = ifelse(open_facilities[i, k], new_cjk, cjk)
    end
    customer_choice_costs[j, k] = cjk
end

function gpu_total_cost_coarse!(
    customer_choice_costs::AbstractVector,
    open_facilities::AbstractMatrix{Bool},
    problem::MultipleFacilityLocationProblem,
)
    (; setup_costs, serving_costs) = problem
    backend = get_backend(problem)
    kernel! = choose_facilities_coarse!(backend)
    kernel!(
        customer_choice_costs,
        open_facilities,
        serving_costs;
        ndrange=size(customer_choice_costs),
    )
    c1 = dot(open_facilities, setup_costs)
    c2 = sum(customer_choice_costs)
    return c1 + c2
end

function gpu_total_cost!(
    customer_choice_costs::AbstractMatrix,
    open_facilities::AbstractMatrix{Bool},
    problem::MultipleFacilityLocationProblem,
)
    (; setup_costs, serving_costs) = problem
    backend = get_backend(problem)
    kernel! = choose_facilities!(backend)
    kernel!(
        customer_choice_costs,
        open_facilities,
        serving_costs;
        ndrange=size(customer_choice_costs),
    )
    c1 = dot(open_facilities, setup_costs)
    c2 = sum(customer_choice_costs)
    return c1 + c2
end

function gpu_total_cost_coarse(
    open_facilities::AbstractMatrix{Bool}, problem::MultipleFacilityLocationProblem
)
    backend = get_backend(problem)
    customer_choice_costs = allocate(backend, eltype(problem), (nb_instances(problem),))
    return gpu_total_cost_coarse!(customer_choice_costs, open_facilities, problem)
end

function gpu_total_cost(
    open_facilities::AbstractMatrix{Bool}, problem::MultipleFacilityLocationProblem
)
    backend = get_backend(problem)
    customer_choice_costs = allocate(
        backend, eltype(problem), (nb_customers(problem), nb_instances(problem))
    )
    return gpu_total_cost!(customer_choice_costs, open_facilities, problem)
end
