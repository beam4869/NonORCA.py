using CSV
using DataFrames
using Dates
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
mkpath(RESULT_DIR)

function residuals_feasible(result)
    result.z !== nothing &&
    result.primal_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) &&
    result.residuals.max_abs_equality <= 1.0e-6 &&
    result.residuals.max_abs_source_carbon_diagnostic <= 1.0e-6 &&
    result.residuals.max_inequality_violation <= 1.0e-6
end

solver_converged(result) = result.termination in (
    MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED,
)

status_ok(result) = solver_converged(result) && residuals_feasible(result)

function main()
    p = baseline_parameters()
    feasibility = solve_ccus(p; objective=:feasibility, time_limit=120.0)
    results = CCUSSolveResult[feasibility]
    start = status_ok(feasibility) ? feasibility.z : initial_feasible_guess(p)
    for objective in [:TAC, :TotEmiss, :ISI, :MaxCapture]
        result = solve_ccus(p; objective, start, time_limit=120.0)
        push!(results, result)
        println(objective, " | ", result.termination, " | ", result.primal_status,
                " | max_eq=", isnothing(result.residuals) ? missing : result.residuals.max_abs_equality,
                " | max_ineq=", isnothing(result.residuals) ? missing : result.residuals.max_inequality_violation)
    end

    summary_rows = NamedTuple[]
    point_rows = NamedTuple[]
    z_names = Symbol.("z" .* string.(1:82))
    for result in results
        o = result.objective_values
        r = result.residuals
        push!(summary_rows, (
            scenario = "baseline",
            solve_type = String(result.objective),
            termination = string(result.termination),
            primal_status = string(result.primal_status),
            solve_time_s = result.solve_time,
            solver_converged = solver_converged(result),
            residuals_feasible = residuals_feasible(result),
            accepted = status_ok(result),
            TAC = isnothing(o) ? missing : o.TAC,
            TotEmiss = isnothing(o) ? missing : o.TotEmiss,
            ISI = isnothing(o) ? missing : o.ISI,
            NetCapture = isnothing(o) ? missing : o.NetCapture,
            capture_fraction_achieved = isnothing(o) ? missing : o.NetCapture / co2_generation(p),
            max_abs_equality = isnothing(r) ? missing : r.max_abs_equality,
            max_abs_source_carbon_diagnostic = isnothing(r) ? missing : r.max_abs_source_carbon_diagnostic,
            max_inequality_violation = isnothing(r) ? missing : r.max_inequality_violation,
        ))
        if status_ok(result)
            z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
            push!(point_rows, merge((scenario="baseline", point_id=String(result.objective)), z_tuple))
        end
    end

    CSV.write(joinpath(RESULT_DIR, "baseline_solves.csv"), DataFrame(summary_rows))
    CSV.write(joinpath(RESULT_DIR, "baseline_points.csv"), DataFrame(point_rows))

    open(joinpath(RESULT_DIR, "baseline_run.txt"), "w") do io
        println(io, "started_at = ", Dates.now())
        println(io, "julia_version = ", VERSION)
        println(io, "capture_target = ", p.capture_fraction)
        println(io, "co2_generation_kt_y = ", co2_generation(p))
        println(io, "all_required_solves_feasible = ", all(status_ok, results))
    end

    all(status_ok, results) || error("At least one baseline solve failed the feasibility residual checks")
    println("Saved baseline results in ", RESULT_DIR)
end

main()
