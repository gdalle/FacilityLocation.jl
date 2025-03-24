"""
Compute the neighboorhodd by switching the diagonal for each instance.
"""
@kernel function switch_kernel!(Y)
    i, k = @index(Global, NTuple)
    Y[i, i, k] = -Y[i, i, k] + 1
end

"""
Compute the setup cost for each neighboorhood `n` and instance `k`.
"""
@kernel function setup_cost_kernel!(C, @Const(Y), @Const(setup_costs))
    n, k = @index(Global, NTuple)

    I = size(setup_costs, 1)

    T = eltype(C)

    res = zero(T)
    for i in 1:I
        res += setup_costs[i, k] * Y[i, n, k]
    end
    C[n, k] = res
end

"""
Compute serving costs for each neighboorhood `n` and instance `k`.
"""
@kernel function serving_costs_kernel!(C, @Const(Y), @Const(serving_costs))
    n, k = @index(Global, NTuple)

    I = size(serving_costs, 1)
    J = size(serving_costs, 2)

    T = eltype(C)

    res = zero(T)
    for j in 1:J
        local_res = typemax(T)
        for i in 1:I
            if !Y[i, n, k]
                continue
            end
            local_res = min(local_res, serving_costs[i, j, k])
        end
        res += local_res
    end
    C[n, k] = res
end

"""
For each instance `k` store the argmin in `M[k]`` and the min value in `V[k]`.
"""
@kernel function argmin_kernel!(M, V, @Const(C))
    k = @index(Global)

    best_index = -1
    min_value = typemax(eltype(C))

    N = size(C, 1)
    for n in 1:N
        if C[n, k] < min_value
            best_index = n
            min_value = C[n, k]
        end
    end

    M[k] = best_index
    V[k] = min_value
end

"""
For each instance `k`, set all neighboorhoods to the best solution in the argmin `M`.
"""
@kernel function duplicate_best_solution_kernel!(Y, @Const(M))
    i, n, k = @index(Global, NTuple)
    Y[i, n, k] = Y[i, M[k], k]
end

"""
Copy the current best solution to `y`.
"""
@kernel function retrieve_best_solution_kernel!(y, @Const(Y))
    i, k = @index(Global, NTuple)
    N = size(Y, 2)
    y[i, k] = Y[i, N, k]
end

"""
Compute customer assignments from solution `Y` and rank to facility matrix.
"""
@kernel function compute_assignments_kernel!(X, @Const(Y), @Const(rank_to_facility))
    j, k = @index(Global, NTuple)
    I = size(rank_to_facility, 1)
    for r in 1:I
        i = rank_to_facility[r, j, k]
        if Y[i, k]
            X[j, k] = i
            break
        end
    end
end

"""
    gpu_local_search(problem::FLP; iterations=10)

Perform a local search on the (gpu) backend of the problem.
"""
function gpu_local_search(problem::FLP; iterations=10, verbose=false)
    backend = get_backend(problem)

    I = nb_facilities(problem)
    J = nb_customers(problem)
    K = nb_instances(problem)
    N = I + 1 # last solution is the current best one

    (; setup_costs, serving_costs) = problem
    T = eltype(setup_costs)

    Y = KernelAbstractions.ones(backend, Bool, I, N, K)
    C = KernelAbstractions.zeros(backend, T, N, K)
    CC = KernelAbstractions.zeros(backend, T, N, K)
    M = KernelAbstractions.zeros(backend, Int, K)
    V = KernelAbstractions.zeros(backend, T, K)

    for it in 1:iterations
        switch_kernel!(backend)(Y; ndrange=(I, K))
        synchronize(backend)
        setup_cost_kernel!(backend)(CC, Y, setup_costs; ndrange=(N, K))
        synchronize(backend)
        serving_costs_kernel!(backend)(C, Y, serving_costs; ndrange=(N, K))
        synchronize(backend)
        argmin_kernel!(backend)(M, V, C + CC; ndrange=K)
        synchronize(backend)
        duplicate_best_solution_kernel!(backend)(Y, M; ndrange=(I, N, K))
        synchronize(backend)

        verbose && println("Iteration $it: $(sum(V))")

        if all(M .== N)
            break
        end
    end

    best_y = KernelAbstractions.ones(backend, Bool, I, K)
    retrieve_best_solution_kernel!(backend)(best_y, Y; ndrange=(I, K))
    synchronize(backend)

    assignments = KernelAbstractions.zeros(backend, Int, J, K)
    compute_assignments_kernel!(backend)(
        assignments, best_y, problem.rank_to_facility; ndrange=(J, K)
    )
    synchronize(backend)

    solution = Solution(best_y, assignments)

    return solution, V
end
