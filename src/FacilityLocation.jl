module FacilityLocation

using Adapt
using Base.Threads
using FacilityLocationProblems
using GPUArrays
using KernelAbstractions:
    KernelAbstractions, CPU, adapt, get_backend, @kernel, @index, @Const, synchronize
using LinearAlgebra
using OhMyThreads
using Random: AbstractRNG
using StableRNGs

const KA = KernelAbstractions

include("problem.jl")
include("solution.jl")
include("cpu.jl")
include("gpu_new.jl")

function plot_instance end
function plot_solution end

export FacilityLocationProblem
export nb_instances, nb_facilities, nb_customers, instances, facilities, customers
export Solution, total_cost, local_search
export GPUSolution, gpu_local_search

export plot_instance, plot_solution

end # module FacilityLocation
