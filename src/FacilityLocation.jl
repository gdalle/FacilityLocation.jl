module FacilityLocation

using FacilityLocationProblems
using GPUArrays
using LinearAlgebra
using KernelAbstractions
using Metal: MetalBackend

include("types.jl")
include("sequential.jl")
include("parallel.jl")
include("utils.jl")

export MultipleFacilityLocationProblem
export total_cost
export gpu_total_cost, gpu_total_cost!
export gpu

end # module FacilityLocation
