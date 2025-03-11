module FacilityLocation

using FacilityLocationProblems: FacilityLocationProblem, loadFacilityLocationProblem
using GPUArrays: AbstractGPUVector, AbstractGPUMatrix
using LinearAlgebra: dot
using KernelAbstractions: Backend, allocate
using Metal: MetalBackend

export GPUFacilityLocationProblem
export total_cost

struct GPUFacilityLocationProblem{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    name::String
    fixed_costs::V
    costs::M
end

function GPUFacilityLocationProblem(problem::FacilityLocationProblem, backend::Backend)
    (; name, fixed_costs, costs) = problem
    gpu_fixed_costs = allocate(backend, Float32, size(fixed_costs)...)
    gpu_costs = allocate(backend, Float32, size(costs)...)
    copyto!(gpu_fixed_costs, fixed_costs)
    copyto!(gpu_costs, costs)
    return GPUFacilityLocationProblem(name, gpu_fixed_costs, gpu_costs)
end

function total_cost(active::AbstractVector{Bool}, problem::FacilityLocationProblem)
    (; fixed_costs, costs) = problem
    c = 0.0
    for i in eachindex(fixed_costs)
        if active[i]
            c += fixed_costs[i]
        end
    end
    for j in axes(costs, 2)
        cj = typemax(eltype(costs))
        for i in axes(costs, 1)
            cij = costs[i, j]
            if active[i]
                cj = min(cj, cij)
            end
        end
        c += cj
    end
    return c
end

function total_cost(active::AbstractVector{Bool}, gpu_problem::GPUFacilityLocationProblem)
    (; fixed_costs, costs) = gpu_problem
    c = dot(active, fixed_costs)
    for j in axes(costs, 2)
        c += mapreduce(min, active, view(costs, :, j)) do a, c
            ifelse(a, c, typemax(c))
        end
    end
    return c
end

end # module FacilityLocation
