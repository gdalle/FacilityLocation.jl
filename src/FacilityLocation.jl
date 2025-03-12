module FacilityLocation

using Base.Threads
using FacilityLocationProblems
using GPUArrays
using LinearAlgebra
using KernelAbstractions
using Metal: MetalBackend
using OhMyThreads

include("types.jl")
include("cpu.jl")
include("gpu.jl")

export MultipleFacilityLocationProblem
export total_cost
export gpu_total_cost, gpu_total_cost!
export assign_customers!, evaluate_addition!, evaluate_deletion!
export local_search

end # module FacilityLocation
