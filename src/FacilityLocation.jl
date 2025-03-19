module FacilityLocation

using Adapt
using Base.Threads
using FacilityLocationProblems
using GPUArrays
using LinearAlgebra
using KernelAbstractions
using OhMyThreads

include("problem.jl")
include("cpu.jl")
# include("gpu.jl")

export FacilityLocationProblem
export nb_instances, nb_facilities, nb_customers, instances, facilities, customers
export Solution, total_cost, local_search

end # module FacilityLocation
