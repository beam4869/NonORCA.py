using CSV
using DataFrames
using Dates
using ForwardDiff
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const CONVERGED = (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
const PRIMAL_OK = (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)

solver_converged(result) = result.termination in CONVERGED

function residuals_feasible(result; tolerance=1.0e-6)
    result.z !== nothing &&
    result.primal_status in PRIMAL_OK &&
    result.residuals.max_abs_equality <= tolerance &&
    result.residuals.max_abs_source_carbon_diagnostic <= tolerance &&
    result.residuals.max_inequality_violation <= tolerance
end

accepted(result) = solver_converged(result) && residuals_feasible(result)

function scenario_specs()
    return [
        (name="baseline", parameter="baseline", value=1.0, p=baseline_parameters()),
        (name="capture_low", parameter="capture_fraction", value=0.20,
         p=baseline_parameters(capture_fraction=0.20)),
        (name="capture_high", parameter="capture_fraction", value=0.60,
         p=baseline_parameters(capture_fraction=0.60)),
        (name="capture_near_limit", parameter="capture_fraction", value=0.85,
         p=baseline_parameters(capture_fraction=0.85)),
        (name="capture_infeasible", parameter="capture_fraction", value=0.90,
         p=baseline_parameters(capture_fraction=0.90)),
        (name="efficiency_low", parameter="capture_efficiency", value=0.80,
         p=baseline_parameters(capture_efficiency=0.80)),
        (name="efficiency_high", parameter="capture_efficiency", value=0.95,
         p=baseline_parameters(capture_efficiency=0.95)),
        (name="grid_low", parameter="power_carbon_intensity", value=0.183,
         p=baseline_parameters(power_carbon_intensity=0.183)),
        (name="grid_high", parameter="power_carbon_intensity", value=0.732,
         p=baseline_parameters(power_carbon_intensity=0.732)),
        (name="power_price_high", parameter="power_price", value=0.04,
         p=baseline_parameters(power_price=0.04)),
        (name="capacity_low", parameter="sink_capacity_scale", value=0.80,
         p=baseline_parameters(sink_capacity_scale=0.80)),
        (name="sink_loss_high", parameter="sink_loss_scale", value=1.50,
         p=baseline_parameters(sink_loss_scale=1.50)),
    ]
end

function solve_summary_row(spec, result)
    o = result.objective_values
    r = result.residuals
    return (
        scenario=spec.name,
        varied_parameter=spec.parameter,
        parameter_value=spec.value,
        solve_type=String(result.objective),
        termination=string(result.termination),
        primal_status=string(result.primal_status),
        solve_time_s=result.solve_time,
        solver_converged=solver_converged(result),
        residuals_feasible=residuals_feasible(result),
        accepted=accepted(result),
        TAC=isnothing(o) ? missing : o.TAC,
        TotEmiss=isnothing(o) ? missing : o.TotEmiss,
        ISI=isnothing(o) ? missing : o.ISI,
        NetCapture=isnothing(o) ? missing : o.NetCapture,
        capture_fraction_achieved=isnothing(o) ? missing : o.NetCapture / co2_generation(spec.p),
        max_abs_equality=isnothing(r) ? missing : r.max_abs_equality,
        max_abs_source_carbon_diagnostic=isnothing(r) ? missing : r.max_abs_source_carbon_diagnostic,
        max_inequality_violation=isnothing(r) ? missing : r.max_inequality_violation,
    )
end

function add_gradient_rows!(objective_rows, equality_rows, inequality_rows, spec, result, gradient_names)
    z = result.z
    p = spec.p
    point_id = spec.name * "__" * String(result.objective)
    Jobj = ForwardDiff.jacobian(x -> objective_vector(x, p), z)
    Jeq = ForwardDiff.jacobian(x -> equality_values(x, p), z)
    Jineq = ForwardDiff.jacobian(x -> inequality_values(x, p), z)
    eq_values = equality_values(z, p)
    ineq_values = inequality_values(z, p)
    grad_tuple(v) = NamedTuple{Tuple(gradient_names)}(Tuple(Float64.(v)))

    for i in eachindex(objective_names())
        push!(objective_rows, merge(
            (scenario=spec.name, point_id, objective=objective_names()[i]),
            grad_tuple(Jobj[i, :]),
        ))
    end
    for i in eachindex(equality_names())
        push!(equality_rows, merge(
            (scenario=spec.name, point_id, constraint=equality_names()[i], residual=eq_values[i]),
            grad_tuple(Jeq[i, :]),
        ))
    end
    for i in eachindex(inequality_names())
        push!(inequality_rows, merge(
            (scenario=spec.name, point_id, constraint=inequality_names()[i],
             value=ineq_values[i], active_1e6=ineq_values[i] >= -1.0e-6),
            grad_tuple(Jineq[i, :]),
        ))
    end
end

function main()
    mkpath(RESULT_DIR)
    solve_rows = NamedTuple[]
    point_rows = NamedTuple[]
    objective_rows = NamedTuple[]
    equality_rows = NamedTuple[]
    inequality_rows = NamedTuple[]
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))
    gradient_names = Symbol.("dz" .* string.(eachindex(variable_names())))

    for spec in scenario_specs()
        feasibility = solve_ccus(spec.p; objective=:feasibility, time_limit=60.0)
        results = CCUSSolveResult[feasibility]
        if accepted(feasibility)
            for objective in (:TAC, :TotEmiss, :ISI, :MaxCapture)
                push!(results, solve_ccus(spec.p; objective, start=feasibility.z, time_limit=60.0))
            end
        end

        for result in results
            push!(solve_rows, solve_summary_row(spec, result))
            if accepted(result) && result.objective != :feasibility
                point_id = spec.name * "__" * String(result.objective)
                o = result.objective_values
                z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
                push!(point_rows, merge((
                    scenario=spec.name,
                    varied_parameter=spec.parameter,
                    parameter_value=spec.value,
                    point_id,
                    solve_type=String(result.objective),
                    TAC=o.TAC,
                    TotEmiss=o.TotEmiss,
                    ISI=o.ISI,
                    NetCapture=o.NetCapture,
                    capture_fraction_achieved=o.NetCapture / co2_generation(spec.p),
                ), z_tuple))
                add_gradient_rows!(objective_rows, equality_rows, inequality_rows,
                                   spec, result, gradient_names)
            end
        end

        accepted_count = count(accepted, results)
        println(spec.name, ": ", accepted_count, "/", length(results), " accepted")
    end

    CSV.write(joinpath(RESULT_DIR, "scenario_solves.csv"), DataFrame(solve_rows))
    CSV.write(joinpath(RESULT_DIR, "selected_points.csv"), DataFrame(point_rows))
    CSV.write(joinpath(RESULT_DIR, "objective_gradients.csv"), DataFrame(objective_rows))
    CSV.write(joinpath(RESULT_DIR, "equality_gradients.csv"), DataFrame(equality_rows))
    CSV.write(joinpath(RESULT_DIR, "inequality_gradients.csv"), DataFrame(inequality_rows))
    open(joinpath(RESULT_DIR, "scenario_run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "julia_version = ", VERSION)
        println(io, "number_of_scenarios = ", length(scenario_specs()))
        println(io, "number_of_selected_points = ", length(point_rows))
        println(io, "active_inequality_tolerance = 1e-6")
    end
end

main()
