using CSV
using DataFrames
using Dates
using JuMP
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const CONVERGED = (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
const PRIMAL_OK = (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
const AUGMENTATION = 1.0e-3
const FEASIBILITY_TOLERANCE = 1.0e-6
const PARETO_SOLVE_TIME_LIMIT = parse(
    Float64, get(ENV, "CCUS_PARETO_SOLVE_TIME_LIMIT", "30.0")
)
const PARETO_PROFILE = get(ENV, "CCUS_PARETO_PROFILE", "baseline")
const OUTPUT_PREFIX = PARETO_PROFILE == "baseline" ? "" : PARETO_PROFILE * "_"

function pareto_parameters()
    PARETO_PROFILE == "baseline" && return baseline_parameters()
    PARETO_PROFILE == "direct_use_expansion" && return direct_use_expansion_parameters()
    PARETO_PROFILE == "supply_chain_condition1" && return supply_chain_condition1_parameters()
    PARETO_PROFILE == "supply_chain_condition2" && return supply_chain_condition2_parameters()
    error("Unknown CCUS_PARETO_PROFILE: $PARETO_PROFILE")
end

result_path(filename) = joinpath(RESULT_DIR, OUTPUT_PREFIX * filename)

normalized_objectives(o, lower, ranges) = [
    (o.TAC - lower[1]) / ranges[1],
    (o.TotEmiss - lower[2]) / ranges[2],
    (o.ISI - lower[3]) / ranges[3],
]

function base_feasible(z, p)
    r = residual_summary(z, p)
    r.max_abs_equality <= FEASIBILITY_TOLERANCE &&
    r.max_abs_source_carbon_diagnostic <= FEASIBILITY_TOLERANCE &&
    r.max_inequality_violation <= FEASIBILITY_TOLERANCE
end

function solve_scalarization(
    p,
    lower,
    ranges;
    mode::Symbol,
    epsilon_a::Float64,
    epsilon_b::Union{Nothing, Float64}=nothing,
    start,
    time_limit=30.0,
)
    bm = build_model(p; objective=:feasibility, start, time_limit)
    set_optimizer_attribute(bm.model, "max_iter", 8000)
    tac = bm.tac
    emissions = bm.total_emissions
    isi = bm.isi

    if mode in (:full, :full_TAC)
        @constraint(bm.model, (emissions - epsilon_a) / ranges[2] <= 0)
        @constraint(bm.model, (isi - epsilon_b) / ranges[3] <= 0)
        @NLobjective(bm.model, Min,
            (tac - lower[1]) / ranges[1] + AUGMENTATION *
            ((emissions - lower[2]) / ranges[2] + (isi - lower[3]) / ranges[3]))
    elseif mode == :full_TotEmiss
        @NLconstraint(bm.model, (tac - epsilon_a) / ranges[1] <= 0)
        @constraint(bm.model, (isi - epsilon_b) / ranges[3] <= 0)
        @NLobjective(bm.model, Min,
            (emissions - lower[2]) / ranges[2] + AUGMENTATION *
            ((tac - lower[1]) / ranges[1] + (isi - lower[3]) / ranges[3]))
    elseif mode == :full_ISI
        @NLconstraint(bm.model, (tac - epsilon_a) / ranges[1] <= 0)
        @constraint(bm.model, (emissions - epsilon_b) / ranges[2] <= 0)
        @NLobjective(bm.model, Min,
            (isi - lower[3]) / ranges[3] + AUGMENTATION *
            ((tac - lower[1]) / ranges[1] + (emissions - lower[2]) / ranges[2]))
    elseif mode == :group_TAC_TotEmiss
        @constraint(bm.model, (isi - epsilon_a) / ranges[3] <= 0)
        @NLobjective(bm.model, Min,
            (tac - lower[1]) / ranges[1] + (emissions - lower[2]) / ranges[2] +
            AUGMENTATION * (isi - lower[3]) / ranges[3])
    elseif mode == :group_TAC_ISI
        @constraint(bm.model, (emissions - epsilon_a) / ranges[2] <= 0)
        @NLobjective(bm.model, Min,
            (tac - lower[1]) / ranges[1] + (isi - lower[3]) / ranges[3] +
            AUGMENTATION * (emissions - lower[2]) / ranges[2])
    elseif mode == :group_TotEmiss_ISI
        @NLconstraint(bm.model, (tac - epsilon_a) / ranges[1] <= 0)
        @NLobjective(bm.model, Min,
            (emissions - lower[2]) / ranges[2] + (isi - lower[3]) / ranges[3] +
            AUGMENTATION * (tac - lower[1]) / ranges[1])
    else
        error("Unknown scalarization mode: $mode")
    end

    elapsed = @elapsed optimize!(bm.model)
    termination = termination_status(bm.model)
    primal = primal_status(bm.model)
    if !has_values(bm.model)
        return (; accepted=false, termination, primal, raw_status=raw_status(bm.model),
                solve_time=elapsed, z=nothing, objectives=nothing, residuals=nothing,
                scalarized=Inf, epsilon_violation=Inf)
    end

    z = value.(bm.z)
    o = objective_values(z, p)
    r = residual_summary(z, p)
    norm = normalized_objectives(o, lower, ranges)
    if mode in (:full, :full_TAC)
        epsilon_violation = max(
            (o.TotEmiss - epsilon_a) / ranges[2],
            (o.ISI - epsilon_b) / ranges[3],
            0.0,
        )
        scalarized = norm[1] + AUGMENTATION * (norm[2] + norm[3])
    elseif mode == :full_TotEmiss
        epsilon_violation = max(
            (o.TAC - epsilon_a) / ranges[1],
            (o.ISI - epsilon_b) / ranges[3],
            0.0,
        )
        scalarized = norm[2] + AUGMENTATION * (norm[1] + norm[3])
    elseif mode == :full_ISI
        epsilon_violation = max(
            (o.TAC - epsilon_a) / ranges[1],
            (o.TotEmiss - epsilon_b) / ranges[2],
            0.0,
        )
        scalarized = norm[3] + AUGMENTATION * (norm[1] + norm[2])
    elseif mode == :group_TAC_TotEmiss
        epsilon_violation = max((o.ISI - epsilon_a) / ranges[3], 0.0)
        scalarized = norm[1] + norm[2] + AUGMENTATION * norm[3]
    elseif mode == :group_TAC_ISI
        epsilon_violation = max((o.TotEmiss - epsilon_a) / ranges[2], 0.0)
        scalarized = norm[1] + norm[3] + AUGMENTATION * norm[2]
    else
        epsilon_violation = max((o.TAC - epsilon_a) / ranges[1], 0.0)
        scalarized = norm[2] + norm[3] + AUGMENTATION * norm[1]
    end
    accepted = termination in CONVERGED && primal in PRIMAL_OK && base_feasible(z, p) &&
               epsilon_violation <= FEASIBILITY_TOLERANCE
    return (; accepted, termination, primal, raw_status=raw_status(bm.model),
            solve_time=elapsed, z, objectives=o, residuals=r, scalarized,
            epsilon_violation)
end

function distinct_starts(starts)
    unique_starts = Vector{Vector{Float64}}()
    for start in starts
        isnothing(start) && continue
        if all(maximum(abs.(start .- existing)) > 1.0e-8 for existing in unique_starts)
            push!(unique_starts, Float64.(start))
        end
    end
    return unique_starts
end

function target_violation(o, ranges, mode, epsilon_a, epsilon_b)
    if mode in (:full, :full_TAC)
        return max(0.0, (o.TotEmiss - epsilon_a) / ranges[2]) +
               max(0.0, (o.ISI - epsilon_b) / ranges[3])
    elseif mode == :full_TotEmiss
        return max(0.0, (o.TAC - epsilon_a) / ranges[1]) +
               max(0.0, (o.ISI - epsilon_b) / ranges[3])
    elseif mode == :full_ISI
        return max(0.0, (o.TAC - epsilon_a) / ranges[1]) +
               max(0.0, (o.TotEmiss - epsilon_b) / ranges[2])
    elseif mode == :group_TAC_TotEmiss
        return max(0.0, (o.ISI - epsilon_a) / ranges[3])
    elseif mode == :group_TAC_ISI
        return max(0.0, (o.TotEmiss - epsilon_a) / ranges[2])
    else
        return max(0.0, (o.TAC - epsilon_a) / ranges[1])
    end
end

function solve_with_deterministic_starts(
    p,
    lower,
    ranges,
    anchors;
    mode,
    epsilon_a,
    epsilon_b=nothing,
    warm_starts=Vector{Vector{Float64}}(),
)
    ordered_anchors = sort(
        anchors,
        by=a -> (target_violation(a.objectives, ranges, mode, epsilon_a, epsilon_b),
                 sum(abs, normalized_objectives(a.objectives, lower, ranges))),
    )
    starts = distinct_starts(vcat(warm_starts, [a.z for a in ordered_anchors]))
    candidates = NamedTuple[]
    attempts = 0
    initial_attempts = min(3, length(starts))
    for start in starts[1:initial_attempts]
        attempts += 1
        result = solve_scalarization(
            p, lower, ranges;
            mode, epsilon_a, epsilon_b, start,
            time_limit=PARETO_SOLVE_TIME_LIMIT,
        )
        result.accepted && push!(candidates, result)
    end
    if isempty(candidates)
        for start in starts[(initial_attempts + 1):end]
            attempts += 1
            result = solve_scalarization(
                p, lower, ranges;
                mode, epsilon_a, epsilon_b, start,
                time_limit=PARETO_SOLVE_TIME_LIMIT,
            )
            result.accepted && push!(candidates, result)
        end
    end
    isempty(candidates) && return (; best=nothing, attempts, successful=0)
    best = candidates[argmin(getfield.(candidates, :scalarized))]
    return (; best, attempts, successful=length(candidates))
end

function result_row(method, index_a, index_b, epsilon_tac, epsilon_emissions,
                    epsilon_isi, result, attempts, successful, z_names)
    o = result.objectives
    r = result.residuals
    z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
    return merge((
        method=String(method),
        index_a,
        index_b,
        epsilon_TAC=epsilon_tac,
        epsilon_TotEmiss=epsilon_emissions,
        epsilon_ISI=epsilon_isi,
        attempts,
        successful_starts=successful,
        termination=string(result.termination),
        raw_status=result.raw_status,
        solve_time_s=result.solve_time,
        scalarized_objective=result.scalarized,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        NetCapture=o.NetCapture,
        max_abs_equality=r.max_abs_equality,
        max_abs_source_carbon_diagnostic=r.max_abs_source_carbon_diagnostic,
        max_inequality_violation=r.max_inequality_violation,
        epsilon_violation=result.epsilon_violation,
    ), z_tuple)
end

function anchor_data(p)
    feasibility = solve_ccus(p; objective=:feasibility, time_limit=60.0)
    isnothing(feasibility.z) && error("Baseline feasibility solve did not return values")
    anchors = NamedTuple[]
    for objective in (:TAC, :TotEmiss, :ISI)
        result = solve_ccus(p; objective, start=feasibility.z, time_limit=60.0)
        result.termination in CONVERGED || error("Anchor solve failed for $objective: $(result.termination)")
        base_feasible(result.z, p) || error("Anchor solution failed residual checks for $objective")
        push!(anchors, (name=String(objective), z=result.z, objectives=result.objective_values))
    end
    push!(anchors, (name="feasibility", z=feasibility.z, objectives=feasibility.objective_values))
    return anchors
end

function main()
    mkpath(RESULT_DIR)
    p = pareto_parameters()
    grid_size_tac = parse(Int, get(ENV, "CCUS_PARETO_GRID_TAC", get(ENV, "CCUS_PARETO_GRID", "21")))
    grid_size_aux = parse(Int, get(ENV, "CCUS_PARETO_GRID_AUX", "15"))
    reduced_size = parse(Int, get(ENV, "CCUS_REDUCED_GRID", "61"))
    grid_size_tac >= 5 || error("CCUS_PARETO_GRID_TAC must be at least 5")
    grid_size_aux >= 5 || error("CCUS_PARETO_GRID_AUX must be at least 5")
    reduced_size >= 5 || error("CCUS_REDUCED_GRID must be at least 5")
    anchors = anchor_data(p)
    payoff = reduce(vcat, [reshape([
        a.objectives.TAC, a.objectives.TotEmiss, a.objectives.ISI
    ], 1, 3) for a in anchors[1:3]])
    lower = [minimum(payoff[:, i]) for i in 1:3]
    upper = [maximum(payoff[:, i]) for i in 1:3]
    ranges = upper .- lower
    any(ranges .<= 0) && error("Degenerate objective range in payoff table")
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))

    anchor_rows = [(
        anchor=a.name,
        TAC=a.objectives.TAC,
        TotEmiss=a.objectives.TotEmiss,
        ISI=a.objectives.ISI,
        NetCapture=a.objectives.NetCapture,
    ) for a in anchors]
    CSV.write(result_path("pareto_payoff_table.csv"), DataFrame(anchor_rows))

    full_rows = NamedTuple[]
    log_rows = NamedTuple[]
    full_specs = [
        (mode=:full_TAC, size=grid_size_tac,
         values_a=collect(range(upper[2], lower[2], length=grid_size_tac)),
         values_b=collect(range(upper[3], lower[3], length=grid_size_tac))),
        (mode=:full_TotEmiss, size=grid_size_aux,
         values_a=collect(range(upper[1], lower[1], length=grid_size_aux)),
         values_b=collect(range(upper[3], lower[3], length=grid_size_aux))),
        (mode=:full_ISI, size=grid_size_aux,
         values_a=collect(range(upper[1], lower[1], length=grid_size_aux)),
         values_b=collect(range(upper[2], lower[2], length=grid_size_aux))),
    ]
    for spec in full_specs
        above = Union{Nothing, Vector{Float64}}[nothing for _ in 1:spec.size]
        accepted_before = length(full_rows)
        for (i, epsilon_a) in enumerate(spec.values_a)
            left = nothing
            current = Union{Nothing, Vector{Float64}}[nothing for _ in 1:spec.size]
            for (j, epsilon_b) in enumerate(spec.values_b)
                warm = distinct_starts([left, above[j]])
                outcome = solve_with_deterministic_starts(
                    p, lower, ranges, anchors;
                    mode=spec.mode, epsilon_a, epsilon_b, warm_starts=warm,
                )
                accepted = !isnothing(outcome.best)
                epsilon_tac = spec.mode == :full_TAC ? missing : epsilon_a
                epsilon_emiss = spec.mode == :full_TAC ? epsilon_a :
                                 spec.mode == :full_ISI ? epsilon_b : missing
                epsilon_isi_value = spec.mode == :full_ISI ? missing : epsilon_b
                push!(log_rows, (
                    method=String(spec.mode), index_a=i, index_b=j,
                    epsilon_TAC=epsilon_tac, epsilon_TotEmiss=epsilon_emiss,
                    epsilon_ISI=epsilon_isi_value, attempts=outcome.attempts,
                    successful_starts=outcome.successful, accepted,
                ))
                if accepted
                    best = outcome.best
                    push!(full_rows, result_row(
                        spec.mode, i, j, epsilon_tac, epsilon_emiss,
                        epsilon_isi_value, best, outcome.attempts,
                        outcome.successful, z_names,
                    ))
                    left = best.z
                    current[j] = best.z
                end
            end
            above = current
            println(spec.mode, " grid row ", i, "/", spec.size,
                    " | accepted=", length(full_rows) - accepted_before)
        end
    end

    reduced_rows = NamedTuple[]
    group_specs = [
        (mode=:group_TAC_TotEmiss, retained=:ISI, lower=lower[3], upper=upper[3]),
        (mode=:group_TAC_ISI, retained=:TotEmiss, lower=lower[2], upper=upper[2]),
        (mode=:group_TotEmiss_ISI, retained=:TAC, lower=lower[1], upper=upper[1]),
    ]
    for spec in group_specs
        warm = nothing
        levels = collect(range(spec.upper, spec.lower, length=reduced_size))
        for (i, epsilon) in enumerate(levels)
            warm_starts = isnothing(warm) ? Vector{Vector{Float64}}() : [warm]
            outcome = solve_with_deterministic_starts(
                p, lower, ranges, anchors;
                mode=spec.mode, epsilon_a=epsilon, warm_starts,
            )
            accepted = !isnothing(outcome.best)
            epsilon_tac = spec.retained == :TAC ? epsilon : missing
            epsilon_emiss = spec.retained == :TotEmiss ? epsilon : missing
            epsilon_isi_value = spec.retained == :ISI ? epsilon : missing
            push!(log_rows, (
                method=String(spec.mode), index_a=i, index_b=missing,
                epsilon_TAC=epsilon_tac, epsilon_TotEmiss=epsilon_emiss,
                epsilon_ISI=epsilon_isi_value, attempts=outcome.attempts,
                successful_starts=outcome.successful, accepted,
            ))
            if accepted
                best = outcome.best
                push!(reduced_rows, result_row(
                    spec.mode, i, missing, epsilon_tac, epsilon_emiss,
                    epsilon_isi_value, best, outcome.attempts,
                    outcome.successful, z_names,
                ))
                warm = best.z
            end
        end
        println(spec.mode, " | accepted=",
                count(row -> row.method == String(spec.mode), reduced_rows),
                "/", reduced_size)
    end

    CSV.write(result_path("pareto_full_raw.csv"), DataFrame(full_rows))
    CSV.write(result_path("pareto_reduced_raw.csv"), DataFrame(reduced_rows))
    CSV.write(result_path("pareto_solve_log.csv"), DataFrame(log_rows))
    open(result_path("pareto_run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "profile = ", PARETO_PROFILE)
        println(io, "method = deterministic augmented epsilon-constraint")
        println(io, "global_optimality = not claimed; Ipopt local NLP solves")
        println(io, "full_grid_size_TAC = ", grid_size_tac)
        println(io, "full_grid_size_TotEmiss = ", grid_size_aux)
        println(io, "full_grid_size_ISI = ", grid_size_aux)
        println(io, "reduced_grid_size = ", reduced_size)
        println(io, "augmentation = ", AUGMENTATION)
        println(io, "per_solve_time_limit_s = ", PARETO_SOLVE_TIME_LIMIT)
        println(io, "deterministic_starts_per_target = 3; all remaining anchors if none succeeds")
        println(io, "accepted_full_points = ", length(full_rows))
        println(io, "accepted_reduced_points = ", length(reduced_rows))
        println(io, "objective_lower = ", lower)
        println(io, "objective_upper_payoff = ", upper)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
