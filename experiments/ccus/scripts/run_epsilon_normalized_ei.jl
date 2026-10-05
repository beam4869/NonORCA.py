using CSV
using DataFrames
using Dates
using JuMP
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const EMISSIONS_WEIGHT = 0.50
const ISI_WEIGHT = 0.50
const AUGMENTATION = 1.0e-4
const FEASIBILITY_TOLERANCE = 1.0e-6
const CONVERGED = (
    MOI.OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_OPTIMAL,
    MOI.ALMOST_LOCALLY_SOLVED,
)
const PRIMAL_OK = (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
const GRID_SIZE = parse(Int, get(ENV, "CCUS_EPSILON_GRID", "121"))
const STARTS_PER_EPSILON = parse(Int, get(ENV, "CCUS_EPSILON_STARTS", "6"))

normalized_grouped_objective(o, lower, ranges) =
    EMISSIONS_WEIGHT * (o.TotEmiss - lower[2]) / ranges[2] +
    ISI_WEIGHT * (o.ISI - lower[3]) / ranges[3]

function base_feasible(z, p)
    r = residual_summary(z, p)
    return r.max_abs_equality <= FEASIBILITY_TOLERANCE &&
           r.max_abs_source_carbon_diagnostic <= FEASIBILITY_TOLERANCE &&
           r.max_inequality_violation <= FEASIBILITY_TOLERANCE
end

function distinct_starts(starts)
    output = Vector{Vector{Float64}}()
    for start in starts
        isnothing(start) && continue
        candidate = Float64.(start)
        if all(maximum(abs.(candidate .- existing)) > 1.0e-8 for existing in output)
            push!(output, candidate)
        end
    end
    return output
end

function solve_once(
    p;
    mode::Symbol,
    start,
    epsilon_tac::Float64=0.0,
    tac_lower::Float64=0.0,
    tac_range::Float64=1.0,
    objective_lower::Vector{Float64}=zeros(3),
    objective_ranges::Vector{Float64}=ones(3),
)
    bm = build_model(p; objective=:feasibility, start, time_limit=60.0)
    set_optimizer_attribute(bm.model, "max_iter", 8000)
    if mode == :TAC
        @NLobjective(bm.model, Min, bm.tac / 1.0e7)
    elseif mode == :grouped
        @objective(
            bm.model,
            Min,
            EMISSIONS_WEIGHT * (bm.total_emissions - objective_lower[2]) /
            objective_ranges[2] +
            ISI_WEIGHT * (bm.isi - objective_lower[3]) / objective_ranges[3],
        )
    elseif mode == :epsilon
        @NLconstraint(bm.model, (bm.tac - epsilon_tac) / tac_range <= 0.0)
        @NLobjective(
            bm.model,
            Min,
            EMISSIONS_WEIGHT * (bm.total_emissions - objective_lower[2]) /
            objective_ranges[2] +
            ISI_WEIGHT * (bm.isi - objective_lower[3]) / objective_ranges[3] +
            AUGMENTATION * (bm.tac - tac_lower) / tac_range,
        )
    else
        error("Unknown solve mode: $mode")
    end

    elapsed = @elapsed optimize!(bm.model)
    termination = termination_status(bm.model)
    primal = primal_status(bm.model)
    if !has_values(bm.model)
        return (;
            accepted=false,
            termination,
            primal,
            raw_status=raw_status(bm.model),
            solve_time=elapsed,
            z=nothing,
            objectives=nothing,
            residuals=nothing,
            grouped=Inf,
            scalarized=Inf,
            epsilon_violation=Inf,
            solution_source="optimized",
        )
    end

    z = value.(bm.z)
    o = objective_values(z, p)
    r = residual_summary(z, p)
    grouped = normalized_grouped_objective(o, objective_lower, objective_ranges)
    scalarized = mode == :TAC ? o.TAC / 1.0e7 :
                 mode == :grouped ? grouped :
                 grouped + AUGMENTATION * (o.TAC - tac_lower) / tac_range
    epsilon_violation = mode == :epsilon ?
        max(0.0, (o.TAC - epsilon_tac) / tac_range) : 0.0
    accepted = termination in CONVERGED && primal in PRIMAL_OK &&
               base_feasible(z, p) && epsilon_violation <= FEASIBILITY_TOLERANCE
    return (;
        accepted,
        termination,
        primal,
        raw_status=raw_status(bm.model),
        solve_time=elapsed,
        z,
        objectives=o,
        residuals=r,
        grouped,
        scalarized,
        epsilon_violation,
        solution_source="optimized",
    )
end

function evaluate_seed(
    z,
    p,
    epsilon_tac,
    tac_lower,
    tac_range,
    objective_lower,
    objective_ranges,
)
    base_feasible(z, p) || return nothing
    o = objective_values(z, p)
    epsilon_violation = max(0.0, (o.TAC - epsilon_tac) / tac_range)
    epsilon_violation <= FEASIBILITY_TOLERANCE || return nothing
    r = residual_summary(z, p)
    grouped = normalized_grouped_objective(o, objective_lower, objective_ranges)
    scalarized = grouped + AUGMENTATION * (o.TAC - tac_lower) / tac_range
    return (;
        accepted=true,
        termination=MOI.OPTIMAL,
        primal=MOI.FEASIBLE_POINT,
        raw_status="Retained feasible multistart seed",
        solve_time=0.0,
        z=Float64.(z),
        objectives=o,
        residuals=r,
        grouped,
        scalarized,
        epsilon_violation,
        solution_source="retained_seed",
    )
end

function best_from_starts(p, starts; kwargs...)
    accepted_results = NamedTuple[]
    attempts = 0
    for start in distinct_starts(starts)
        attempts += 1
        result = solve_once(p; start, kwargs...)
        result.accepted && push!(accepted_results, result)
    end
    isempty(accepted_results) && return (; best=nothing, attempts, successful=0)
    best = accepted_results[argmin(getfield.(accepted_results, :scalarized))]
    return (; best, attempts, successful=length(accepted_results))
end

function best_feasible_seed(
    starts,
    p,
    epsilon_tac,
    tac_lower,
    tac_range,
    objective_lower,
    objective_ranges,
)
    feasible = NamedTuple[]
    for start in distinct_starts(starts)
        result = evaluate_seed(
            start,
            p,
            epsilon_tac,
            tac_lower,
            tac_range,
            objective_lower,
            objective_ranges,
        )
        !isnothing(result) && push!(feasible, result)
    end
    isempty(feasible) && return nothing
    return feasible[argmin(getfield.(feasible, :scalarized))]
end

function dataframe_start(row, z_columns)
    return Float64[row[column] for column in z_columns]
end

function global_seed_starts(candidate_data, z_columns, feasibility_start, lower, ranges)
    grouped = EMISSIONS_WEIGHT .* (candidate_data.TotEmiss .- lower[2]) ./ ranges[2] .+
              ISI_WEIGHT .* (candidate_data.ISI .- lower[3]) ./ ranges[3]
    indices = Int[
        argmin(candidate_data.TAC),
        argmin(grouped),
        argmin(candidate_data.TotEmiss),
        argmin(candidate_data.ISI),
    ]
    ordered = sortperm(candidate_data.TAC)
    for fraction in range(0.0, 1.0, length=9)
        position = clamp(round(Int, 1 + fraction * (length(ordered) - 1)), 1, length(ordered))
        push!(indices, ordered[position])
    end
    return distinct_starts(vcat(
        [feasibility_start],
        [dataframe_start(candidate_data[index, :], z_columns) for index in unique(indices)],
    ))
end

function ranked_candidate_starts(
    candidate_data,
    z_columns,
    epsilon_tac,
    tac_lower,
    tac_range,
    objective_lower,
    objective_ranges,
)
    grouped = EMISSIONS_WEIGHT .* (candidate_data.TotEmiss .- objective_lower[2]) ./
              objective_ranges[2] .+
              ISI_WEIGHT .* (candidate_data.ISI .- objective_lower[3]) ./
              objective_ranges[3]
    tac_normalized = (candidate_data.TAC .- tac_lower) ./ tac_range
    score = grouped .+ AUGMENTATION .* tac_normalized
    feasible = findall(candidate_data.TAC .<= epsilon_tac + FEASIBILITY_TOLERANCE * tac_range)
    indices = isempty(feasible) ? Int[] : feasible[sortperm(score[feasible])]
    if length(indices) < STARTS_PER_EPSILON
        closest = sortperm(abs.(candidate_data.TAC .- epsilon_tac))
        append!(indices, closest)
    end
    indices = unique(indices)[1:min(STARTS_PER_EPSILON, length(unique(indices)))]
    return [dataframe_start(candidate_data[index, :], z_columns) for index in indices]
end

function endpoint_row(name, result, attempts, successful, objective_lower, objective_ranges)
    o = result.objectives
    emissions_normalized = (o.TotEmiss - objective_lower[2]) / objective_ranges[2]
    isi_normalized = (o.ISI - objective_lower[3]) / objective_ranges[3]
    return (
        endpoint=name,
        attempts,
        successful_starts=successful,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        TotEmiss_normalized=emissions_normalized,
        ISI_normalized=isi_normalized,
        J_normalized=result.grouped,
        emissions_contribution=EMISSIONS_WEIGHT * emissions_normalized,
        ISI_contribution=ISI_WEIGHT * isi_normalized,
        max_abs_equality=result.residuals.max_abs_equality,
        max_abs_source_carbon_diagnostic=result.residuals.max_abs_source_carbon_diagnostic,
        max_inequality_violation=result.residuals.max_inequality_violation,
    )
end

function result_row(
    epsilon_index,
    epsilon_tac,
    result,
    attempts,
    successful,
    z_names,
    p,
    objective_lower,
    objective_ranges,
    tac_lower,
    tac_range,
    grouped_lower,
    grouped_range,
)
    o = result.objectives
    r = result.residuals
    structure = system_structure(result.z, p)
    z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
    sink_names_all = sink_names()
    active = join(sink_names_all[structure.active_sinks], ";")
    emissions_normalized = (o.TotEmiss - objective_lower[2]) / objective_ranges[2]
    isi_normalized = (o.ISI - objective_lower[3]) / objective_ranges[3]
    row = (
        epsilon_index,
        epsilon_TAC=epsilon_tac,
        epsilon_normalized=(epsilon_tac - tac_lower) / tac_range,
        epsilon_slack=epsilon_tac - o.TAC,
        epsilon_violation=result.epsilon_violation,
        attempts,
        successful_starts=successful,
        solution_source=result.solution_source,
        termination=string(result.termination),
        raw_status=result.raw_status,
        solve_time_s=result.solve_time,
        augmented_objective=result.scalarized,
        augmentation=AUGMENTATION,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        TAC_normalized=(o.TAC - tac_lower) / tac_range,
        TotEmiss_normalized=emissions_normalized,
        ISI_normalized=isi_normalized,
        J_normalized=result.grouped,
        J_outer_normalized=(result.grouped - grouped_lower) / grouped_range,
        emissions_contribution=EMISSIONS_WEIGHT * emissions_normalized,
        ISI_contribution=ISI_WEIGHT * isi_normalized,
        NetCapture=o.NetCapture,
        max_abs_equality=r.max_abs_equality,
        max_abs_source_carbon_diagnostic=r.max_abs_source_carbon_diagnostic,
        max_inequality_violation=r.max_inequality_violation,
        main_sink=sink_names_all[structure.main_sink],
        active_sinks=active,
        pretreated_fraction=structure.pretreated_fraction,
        direct_fraction=structure.direct_fraction,
        sink1_share=structure.sink_share[1],
        sink2_share=structure.sink_share[2],
        sink3_share=structure.sink_share[3],
        sink4_share=structure.sink_share[4],
        sink5_share=structure.sink_share[5],
        sink6_share=structure.sink_share[6],
    )
    return merge(row, z_tuple)
end

function main()
    GRID_SIZE >= 11 || error("CCUS_EPSILON_GRID must be at least 11")
    STARTS_PER_EPSILON >= 3 || error("CCUS_EPSILON_STARTS must be at least 3")
    mkpath(RESULT_DIR)
    p = direct_use_expansion_parameters()
    candidate_data = CSV.read(
        joinpath(RESULT_DIR, "direct_use_expansion_pareto_frontier.csv"),
        DataFrame,
    )
    normalization_data = CSV.read(
        joinpath(RESULT_DIR, "direct_use_expansion_pareto_payoff_table.csv"),
        DataFrame,
    )
    normalization_data = normalization_data[normalization_data.anchor .!= "feasibility", :]
    objective_lower = [
        minimum(normalization_data.TAC),
        minimum(normalization_data.TotEmiss),
        minimum(normalization_data.ISI),
    ]
    objective_upper = [
        maximum(normalization_data.TAC),
        maximum(normalization_data.TotEmiss),
        maximum(normalization_data.ISI),
    ]
    objective_ranges = objective_upper .- objective_lower
    any(objective_ranges .<= 0) && error("Degenerate normalization range")
    z_columns = Symbol.("z" .* string.(1:82))
    z_names = Symbol.("z" .* string.(1:82))

    feasibility = solve_ccus(p; objective=:feasibility, time_limit=60.0)
    isnothing(feasibility.z) && error("Feasibility solve did not return a point")
    base_feasible(feasibility.z, p) || error("Feasibility start failed residual checks")
    global_starts = global_seed_starts(
        candidate_data,
        z_columns,
        feasibility.z,
        objective_lower,
        objective_ranges,
    )

    tac_endpoint = best_from_starts(
        p,
        global_starts;
        mode=:TAC,
        objective_lower,
        objective_ranges,
    )
    isnothing(tac_endpoint.best) && error("TAC endpoint solve failed")
    grouped_endpoint = best_from_starts(
        p,
        global_starts;
        mode=:grouped,
        objective_lower,
        objective_ranges,
    )
    isnothing(grouped_endpoint.best) && error("Grouped endpoint solve failed")

    tac_lower = tac_endpoint.best.objectives.TAC
    tac_upper = grouped_endpoint.best.objectives.TAC
    grouped_lower = grouped_endpoint.best.grouped
    grouped_upper = tac_endpoint.best.grouped
    tac_range = tac_upper - tac_lower
    grouped_range = grouped_upper - grouped_lower
    tac_range > 0 || error("Degenerate TAC payoff range")
    grouped_range > 0 || error("Degenerate grouped-objective payoff range")

    payoff_rows = [
        endpoint_row(
            "Minimum TAC",
            tac_endpoint.best,
            tac_endpoint.attempts,
            tac_endpoint.successful,
            objective_lower,
            objective_ranges,
        ),
        endpoint_row(
            "Minimum J",
            grouped_endpoint.best,
            grouped_endpoint.attempts,
            grouped_endpoint.successful,
            objective_lower,
            objective_ranges,
        ),
    ]
    CSV.write(
        joinpath(RESULT_DIR, "direct_use_expansion_epsilon_normalized_payoff.csv"),
        DataFrame(payoff_rows),
    )

    rows = NamedTuple[]
    previous = nothing
    epsilon_levels = collect(range(tac_lower, tac_upper, length=GRID_SIZE))
    for (index, epsilon_tac) in enumerate(epsilon_levels)
        if index == 1
            outcome = tac_endpoint
        elseif index == length(epsilon_levels)
            outcome = grouped_endpoint
        else
            ranked = ranked_candidate_starts(
                candidate_data,
                z_columns,
                epsilon_tac,
                tac_lower,
                tac_range,
                objective_lower,
                objective_ranges,
            )
            starts = distinct_starts(vcat(
                [previous, tac_endpoint.best.z, grouped_endpoint.best.z],
                ranked,
            ))
            outcome = best_from_starts(
                p,
                starts;
                mode=:epsilon,
                epsilon_tac,
                tac_lower,
                tac_range,
                objective_lower,
                objective_ranges,
            )
            seed_best = best_feasible_seed(
                starts,
                p,
                epsilon_tac,
                tac_lower,
                tac_range,
                objective_lower,
                objective_ranges,
            )
            if !isnothing(seed_best) &&
               (isnothing(outcome.best) || seed_best.scalarized < outcome.best.scalarized)
                outcome = (;
                    best=seed_best,
                    attempts=outcome.attempts,
                    successful=outcome.successful,
                )
            end
        end
        isnothing(outcome.best) && error("No feasible solution at epsilon index $index")
        push!(rows, result_row(
            index,
            epsilon_tac,
            outcome.best,
            outcome.attempts,
            outcome.successful,
            z_names,
            p,
            objective_lower,
            objective_ranges,
            tac_lower,
            tac_range,
            grouped_lower,
            grouped_range,
        ))
        previous = outcome.best.z
        if index == 1 || index % 10 == 0 || index == length(epsilon_levels)
            println(
                "epsilon ", index, "/", length(epsilon_levels),
                " | TAC=", round(outcome.best.objectives.TAC / 1.0e6, digits=4),
                " M | J=", round(outcome.best.grouped, digits=6),
                " | sink=", sink_names()[system_structure(outcome.best.z, p).main_sink],
            )
        end
    end

    raw_path = joinpath(RESULT_DIR, "direct_use_expansion_epsilon_normalized_raw.csv")
    CSV.write(raw_path, DataFrame(rows))
    open(joinpath(RESULT_DIR, "direct_use_expansion_epsilon_normalized_run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "method = augmented epsilon-constraint")
        println(io, "primary_objective = J + rho * normalized TAC")
        println(io, "constraint = TAC <= epsilon_TAC")
        println(io, "J = 0.50 * normalized TotEmiss + 0.50 * normalized ISI")
        println(io, "rho = ", AUGMENTATION)
        println(io, "global_optimality = not claimed; deterministic Ipopt multi-start local NLP solves")
        println(io, "grid_size = ", GRID_SIZE)
        println(io, "candidate_starts_per_epsilon = ", STARTS_PER_EPSILON)
        println(io, "accepted_epsilon_solutions = ", length(rows))
        println(io, "TAC_lower = ", tac_lower)
        println(io, "TAC_upper = ", tac_upper)
        println(io, "J_lower = ", grouped_lower)
        println(io, "J_upper = ", grouped_upper)
        println(io, "objective_normalization_lower = ", objective_lower)
        println(io, "objective_normalization_upper = ", objective_upper)
    end
    println("Epsilon-constraint normalized output: ", raw_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
