function plot_instance(
    problem::FLP{Ti,Tr,<:AbstractArray{Tr,3}}, k::Integer=1; kwargs...
) where {Ti,Tr}
    @assert k in instances(problem)

    facility_x = problem.facility_coordinates[:, 1, k]
    facility_y = problem.facility_coordinates[:, 2, k]
    customer_x = problem.customer_coordinates[:, 1, k]
    customer_y = problem.customer_coordinates[:, 2, k]

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

function plot_solution(
    solution::Solution, problem::FLP{Ti,Tr,<:AbstractArray{Tr,3}}, k::Integer=1; kwargs...
) where {Ti,Tr}
    fig = plot_instance(problem, k; kwargs...)
    open_facilities = solution.open_facilities[:, k]
    customer_assignments = solution.customer_assignments[:, k]

    facility_x = problem.facility_coordinates[:, 1, k]
    facility_y = problem.facility_coordinates[:, 2, k]

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
            [problem.facility_coordinates[i, 1, k], problem.customer_coordinates[j, 1, k]],
            [problem.facility_coordinates[i, 2, k], problem.customer_coordinates[j, 2, k]];
            color=:black,
            label=nothing,
        )
    end

    return fig
end
