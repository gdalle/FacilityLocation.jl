using FacilityLocation
using Test

@testset verbose = true "FacilityLocation" begin
    @testset verbose = true "Correctness" begin
        include("correctness.jl")
    end
end
