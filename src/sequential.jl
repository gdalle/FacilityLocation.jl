function total_cost(open_facilities::AbstractMatrix{Bool}, problem::MFLP{T}) where {T}
    (; facility_costs, customer_costs) = problem
    @assert size(open_facilities) == size(facility_costs)
    c = zero(T)
    for k in instances(problem)
        @simd for i in facilities(problem)
            c += facility_costs[i, k] * open_facilities[i]
        end
        @simd for j in customers(problem)
            cjk = typemax(T)
            @simd for i in facilities(problem)
                cijk = customer_costs[i, j, k]
                cjk = ifelse(open_facilities[i], min(cjk, cijk), cjk)
            end
            c += cjk
        end
    end
    return c
end
