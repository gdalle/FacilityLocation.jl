module FacilityLocation

using Adapt
using Base.Threads
using FacilityLocationProblems
using GPUArrays
using KernelAbstractions
using LinearAlgebra
using OhMyThreads
using StableRNGs

include("problem.jl")
include("solution.jl")
include("cpu.jl")

function plot_instance end
function plot_solution end

export FacilityLocationProblem
export nb_instances, nb_facilities, nb_customers, instances, facilities, customers
export Solution, total_cost, local_search

export plot_instance, plot_solution

end # module FacilityLocation
