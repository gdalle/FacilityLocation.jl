function total_cost(open_facilities::AbstractMatrix{Bool}, problem::MFLP{T}) where {T}
    (; facility_costs, customer_costs) = problem
    @assert size(open_facilities) == size(facility_costs)
    c = zero(T)
    for k in instances(problem)
        for i in facilities(problem)
            if open_facilities[i, k]
                c += facility_costs[i, k]
            end
        end
        for j in customers(problem)
            cjk = typemax(T)
            for i in facilities(problem)
                if open_facilities[i, k]
                    cjk = min(cjk, customer_costs[i, j, k])
                end
            end
            c += cjk
        end
    end
    return c
end

function assign_customers!(
    customer_assignments::AbstractMatrix{Int},
    open_facilities::AbstractMatrix{Bool},
    problem::MFLP{T},
) where {T}
    (; customer_costs) = problem
    for k in instances(problem)
        for j in customers(problem)
            i_chosen = 0
            cjk = typemax(T)
            for i in facilities(problem)
                cijk = customer_costs[i, j, k]
                if open_facilities[i, k] && cijk < cjk
                    cjk = cijk
                    i_chosen = i
                end
            end
            @assert i_chosen > 0
            customer_assignments[j, k] = i_chosen
        end
    end
    return customer_assignments
end

function evaluate_addition!(
    addition_costs::AbstractMatrix{T},
    open_facilities::AbstractMatrix{Bool},
    customer_assignments::AbstractMatrix{Int},
    problem::MFLP,
) where {T}
    (; facility_costs, customer_costs) = problem
    for k in instances(problem)
        for i in facilities(problem)
            if open_facilities[i, k]
                addition_costs[i, k] = typemax(T)
            else
                addition_costs[i, k] = facility_costs[i, k]
            end
        end
        for j in customers(problem)
            i⁻ = customer_assignments[j, k]
            cjk⁻ = customer_costs[i⁻, j, k]
            for i⁺ in facilities(problem)
                if !open_facilities[i⁺, k]
                    cjk⁺ = customer_costs[i⁺, j, k]
                    if cjk⁺ < cjk⁻
                        addition_costs[i⁺, k] -= (cjk⁻ - cjk⁺)
                    end
                end
            end
        end
    end
    return addition_costs
end

function perform_best_addition!(
    addition_costs::AbstractMatrix{T},
    open_facilities::AbstractMatrix{Bool},
    customer_assignments::AbstractMatrix{Int},
    problem::MFLP,
) where {T}
    (; facility_costs, customer_costs) = problem
    move = false
    for k in instances(problem)
        i_add = 0
        ck = zero(T)
        for i in facilities(problem)
            cik = addition_costs[i, k]
            if !open_facilities[i, k] && cik < ck
                i_add = i
                ck = cik
            end
        end
        if ck < zero(T)
            move = true
            @assert i_add > 0
            open_facilities[i_add, k] = true
            for j in customers(problem)
                i⁻ = customer_assignments[j, k]
                cjk⁻ = customer_costs[i⁻, j, k]
                cjk⁺ = customer_costs[i_add, j, k]
                if cjk⁺ < cjk⁻
                    customer_assignments[j, k] = i_add
                end
            end
        end
    end
    return move
end

function evaluate_deletion!(
    deletion_costs::AbstractMatrix{T},
    open_facilities::AbstractMatrix{Bool},
    customer_assignments::AbstractMatrix{Int},
    problem::MFLP{T},
) where {T}
    (; facility_costs, customer_costs) = problem
    for k in instances(problem)
        for i in facilities(problem)
            if open_facilities[i, k]
                deletion_costs[i, k] = -facility_costs[i, k]
            else
                deletion_costs[i, k] = typemax(T)
            end
        end
        for j in customers(problem)
            i⁻ = customer_assignments[j, k]
            cjk⁻ = customer_costs[i⁻, j, k]
            cjk⁺ = typemax(T)
            for i⁺ in facilities(problem)
                if open_facilities[i⁺, k] && i⁺ != i⁻
                    cjk⁺ = min(cjk⁺, customer_costs[i⁺, j, k])
                end
            end
            deletion_costs[i⁻, k] += (cjk⁺ - cjk⁻)
        end
    end
    return deletion_costs
end

function perform_best_deletion!(
    deletion_costs::AbstractMatrix{T},
    open_facilities::AbstractMatrix{Bool},
    customer_assignments::AbstractMatrix{Int},
    problem::MFLP{T},
) where {T}
    (; facility_costs, customer_costs) = problem
    move = false
    for k in instances(problem)
        nb_open = 0
        i_del = 0
        ck = zero(T)
        for i in facilities(problem)
            cik = deletion_costs[i, k]
            if open_facilities[i, k]
                nb_open += 1
                if cik < ck
                    i_del = i
                    ck = cik
                end
            end
        end
        if nb_open >= 2 && ck < zero(T)
            move = true
            open_facilities[i_del, k] = false
            for j in customers(problem)
                if customer_assignments[j, k] == i_del
                    i⁺ = 0
                    cjk⁺ = typemax(T)
                    for i in facilities(problem)
                        cijk = customer_costs[i, j, k]
                        if open_facilities[i, k] && cijk < cjk⁺
                            i⁺ = i
                            cjk⁺ = cijk
                        end
                    end
                    customer_assignments[j, k] = i⁺
                end
            end
        end
    end
    return move
end

function local_search(
    open_facilities::AbstractMatrix{Bool}, problem::MFLP{T}; iterations=10
) where {T}
    open_facilities = copy(open_facilities)
    I, J, K = nb_facilities(problem), nb_customers(problem), nb_instances(problem)
    @assert size(open_facilities) == (I, K)

    customer_assignments = Matrix{Int}(undef, J, K)
    assign_customers!(customer_assignments, open_facilities, problem)

    addition_costs = fill(typemax(T), I, K)
    deletion_costs = fill(typemax(T), I, K)

    for it in 1:iterations
        @info "Iteration $it"

        evaluate_addition!(addition_costs, open_facilities, customer_assignments, problem)
        move_add = perform_best_addition!(
            addition_costs, open_facilities, customer_assignments, problem
        )

        evaluate_deletion!(deletion_costs, open_facilities, customer_assignments, problem)
        move_del = perform_best_deletion!(
            deletion_costs, open_facilities, customer_assignments, problem
        )

        move_add || move_del || break
    end

    return open_facilities
end
