module FacilityLocationPlotsExt

using FacilityLocation
using Plots

const FLP = FacilityLocationProblem

function FacilityLocation.plot_instance(problem::FLP, k::Integer=1; kwargs...)
    @assert problem.facility_coordinates !== nothing
    @assert problem.customer_coordinates !== nothing
    @assert k in FacilityLocation.instances(problem)

    facility_x = first.(problem.facility_coordinates[:, k])
    facility_y = last.(problem.facility_coordinates[:, k])
    customer_x = first.(problem.customer_coordinates[:, k])
    customer_y = last.(problem.customer_coordinates[:, k])

    fig = plot(; xrange=(0, 1), yrange=(0, 1), kwargs...)
    scatter!(
        fig,
        facility_x,
        facility_y;
        label="Facilities",
        markersize=6,
        marker=:square,
        markercolor=:white,
        markerstrokewidth=2,
        markerstrokecolor=:red,
    )
    scatter!(
        fig,
        customer_x,
        customer_y;
        label="Customers",
        markersize=3,
        marker=:circle,
        markercolor=:blue,
    )
    return fig
end

function FacilityLocation.plot_solution(
    solution::Solution, problem::FLP, k::Integer=1; kwargs...
)
    fig = plot_instance(problem, k; kwargs...)
    open_facilities = solution.open_facilities[:, k]
    customer_assignments = solution.customer_assignments[:, k]

    facility_x = first.(problem.facility_coordinates[:, k])
    facility_y = last.(problem.facility_coordinates[:, k])

    scatter!(
        fig,
        facility_x[open_facilities],
        facility_y[open_facilities];
        markershape=:square,
        markercolor=:red,
        markersize=6,
        markerstrokecolor=:red,
        markerstrokewidth=2,
        label=nothing,
    )

    for j in customers(problem)
        i = customer_assignments[j]
        plot!(
            fig,
            [problem.facility_coordinates[i, k][1], problem.customer_coordinates[j, k][1]],
            [problem.facility_coordinates[i, k][2], problem.customer_coordinates[j, k][2]];
            color=:black,
            label=nothing,
        )
    end

    return fig
end

end
