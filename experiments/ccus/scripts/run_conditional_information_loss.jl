using CSV
using DataFrames
using Dates
using JuMP
using Statistics
import MathOptInterface as MOI

include(joinpath(@__DIR__, "run_pareto_epsilon.jl"))

const CONDITIONAL_LABEL = get(ENV, "CCUS_CONDITIONAL_LABEL", "exact")
const CONDITIONAL_LEVELS = parse(Int, get(ENV, "CCUS_CONDITIONAL_LEVELS", "41"))
const CONDITIONAL_RESOLUTIONS = parse.(
    Int,
    split(get(ENV, "CCUS_CONDITIONAL_RESOLUTIONS", "11,21,41"), ","),
)
const LEXICOGRAPHIC_TOLERANCE = 1.0e-7
const RETAINED_TOLERANCE = 1.0e-6
const STARTS_PER_TARGET = 4

conditional_path(name) = result_path("conditional_info_loss_$(CONDITIONAL_LABEL)_$(name)")

objective_index(name::Symbol) = name == :TAC ? 1 : name == :TotEmiss ? 2 : 3
objective_value(o, name::Symbol) = getproperty(o, name)

function add_retained_equality!(bm, retained::Symbol, target::Float64, range::Float64)
    tac = bm.tac
    emissions = bm.total_emissions
    isi = bm.isi
    if retained == :TAC
        @NLconstraint(bm.model, (tac - target) / range == 0)
    elseif retained == :TotEmiss
        @constraint(bm.model, (emissions - target) / range == 0)
    elseif retained == :ISI
        @constraint(bm.model, (isi - target) / range == 0)
    else
        error("Unknown retained objective: $retained")
    end
end

function add_primary_cap!(bm, primary::Symbol, cap::Float64, range::Float64)
    tac = bm.tac
    emissions = bm.total_emissions
    isi = bm.isi
    if primary == :TAC
        @NLconstraint(bm.model, (tac - cap) / range <= 0)
    elseif primary == :TotEmiss
        @constraint(bm.model, (emissions - cap) / range <= 0)
    elseif primary == :ISI
        @constraint(bm.model, (isi - cap) / range <= 0)
    else
        error("Unknown primary objective: $primary")
    end
end

function set_minimum_objective!(bm, objective::Symbol, lower, ranges)
    tac = bm.tac
    emissions = bm.total_emissions
    isi = bm.isi
    if objective == :TAC
        @NLobjective(bm.model, Min, (tac - lower[1]) / ranges[1])
    elseif objective == :TotEmiss
        @objective(bm.model, Min, (emissions - lower[2]) / ranges[2])
    elseif objective == :ISI
        @objective(bm.model, Min, (isi - lower[3]) / ranges[3])
    else
        error("Unknown objective: $objective")
    end
end

function solve_conditional_once(
    p,
    lower,
    ranges;
    retained::Symbol,
    target::Float64,
    objective::Symbol,
    start,
    primary_cap::Union{Nothing, Tuple{Symbol, Float64}}=nothing,
)
    bm = build_model(p; objective=:feasibility, start, time_limit=45.0)
    set_optimizer_attribute(bm.model, "max_iter", 8000)
    retained_index = objective_index(retained)
    add_retained_equality!(bm, retained, target, ranges[retained_index])
    if !isnothing(primary_cap)
        primary, cap = primary_cap
        add_primary_cap!(bm, primary, cap, ranges[objective_index(primary)])
    end
    set_minimum_objective!(bm, objective, lower, ranges)

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
            retained_error=Inf,
            primary_cap_violation=Inf,
        )
    end

    z = value.(bm.z)
    o = objective_values(z, p)
    residuals = residual_summary(z, p)
    retained_error = abs(objective_value(o, retained) - target) / ranges[retained_index]
    primary_cap_violation = if isnothing(primary_cap)
        0.0
    else
        primary, cap = primary_cap
        max(0.0, (objective_value(o, primary) - cap) / ranges[objective_index(primary)])
    end
    accepted = termination in CONVERGED && primal in PRIMAL_OK &&
               base_feasible(z, p) && retained_error <= RETAINED_TOLERANCE &&
               primary_cap_violation <= FEASIBILITY_TOLERANCE
    return (;
        accepted,
        termination,
        primal,
        raw_status=raw_status(bm.model),
        solve_time=elapsed,
        z,
        objectives=o,
        residuals,
        retained_error,
        primary_cap_violation,
    )
end

function solve_stage_one(
    p,
    lower,
    ranges,
    starts;
    retained::Symbol,
    target::Float64,
    primary::Symbol,
    secondary::Symbol,
)
    candidates = NamedTuple[]
    attempts = 0
    initial = min(STARTS_PER_TARGET, length(starts))
    for start in starts[1:initial]
        attempts += 1
        result = solve_conditional_once(
            p, lower, ranges; retained, target, objective=primary, start,
        )
        result.accepted && push!(candidates, result)
    end
    if isempty(candidates)
        for start in starts[(initial + 1):end]
            attempts += 1
            result = solve_conditional_once(
                p, lower, ranges; retained, target, objective=primary, start,
            )
            result.accepted && push!(candidates, result)
        end
    end
    sort!(candidates, by=r -> (
        objective_value(r.objectives, primary),
        objective_value(r.objectives, secondary),
    ))
    return (; candidates, attempts)
end

function solve_lexicographic_endpoint(
    p,
    lower,
    ranges,
    starts;
    retained::Symbol,
    target::Float64,
    primary::Symbol,
    secondary::Symbol,
)
    stage_one = solve_stage_one(
        p, lower, ranges, starts; retained, target, primary, secondary,
    )
    isempty(stage_one.candidates) && return (;
        best=nothing,
        stage_one_attempts=stage_one.attempts,
        stage_one_successes=0,
        stage_two_attempts=0,
        stage_two_successes=0,
        lexicographic_completed=false,
    )

    primary_best = objective_value(stage_one.candidates[1].objectives, primary)
    primary_cap = primary_best + LEXICOGRAPHIC_TOLERANCE * ranges[objective_index(primary)]
    stage_two_candidates = NamedTuple[]
    stage_two_attempts = 0
    for first_stage in stage_one.candidates
        stage_two_attempts += 1
        result = solve_conditional_once(
            p,
            lower,
            ranges;
            retained,
            target,
            objective=secondary,
            start=first_stage.z,
            primary_cap=(primary, primary_cap),
        )
        result.accepted && push!(stage_two_candidates, result)
    end

    if isempty(stage_two_candidates)
        best = stage_one.candidates[1]
        completed = false
    else
        sort!(stage_two_candidates, by=r -> (
            objective_value(r.objectives, secondary),
            objective_value(r.objectives, primary),
        ))
        best = stage_two_candidates[1]
        completed = true
    end
    return (;
        best,
        stage_one_attempts=stage_one.attempts,
        stage_one_successes=length(stage_one.candidates),
        stage_two_attempts,
        stage_two_successes=length(stage_two_candidates),
        lexicographic_completed=completed,
    )
end

function nearest_frontier_starts(frontier, z_names, retained::Symbol, target::Float64; count=4)
    distances = abs.(Float64.(frontier[!, retained]) .- target)
    indices = sortperm(distances)[1:min(count, nrow(frontier))]
    return [Float64.(collect(frontier[index, z_names])) for index in indices]
end

function endpoint_row(
    grouping,
    retained,
    level_index,
    level_fraction,
    target,
    primary,
    secondary,
    outcome,
    dominated_by_sample,
    z_names,
)
    result = outcome.best
    o = result.objectives
    residuals = result.residuals
    z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
    return merge((
        grouping,
        retained=String(retained),
        level_index,
        level_fraction,
        retained_target=target,
        endpoint="min_$(primary)",
        primary=String(primary),
        secondary=String(secondary),
        lexicographic_completed=outcome.lexicographic_completed,
        stage_one_attempts=outcome.stage_one_attempts,
        stage_one_successes=outcome.stage_one_successes,
        stage_two_attempts=outcome.stage_two_attempts,
        stage_two_successes=outcome.stage_two_successes,
        termination=string(result.termination),
        raw_status=result.raw_status,
        solve_time_s=result.solve_time,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        NetCapture=o.NetCapture,
        retained_normalized_error=result.retained_error,
        primary_cap_violation=result.primary_cap_violation,
        max_abs_equality=residuals.max_abs_equality,
        max_abs_source_carbon_diagnostic=residuals.max_abs_source_carbon_diagnostic,
        max_inequality_violation=residuals.max_inequality_violation,
        dominated_by_sample,
    ), z_tuple)
end

function main()
    CONDITIONAL_LEVELS >= 5 || error("CCUS_CONDITIONAL_LEVELS must be at least 5")
    all(r -> r >= 3, CONDITIONAL_RESOLUTIONS) ||
        error("Every conditional resolution must be at least 3")
    all(r -> r <= CONDITIONAL_LEVELS, CONDITIONAL_RESOLUTIONS) ||
        error("Conditional resolutions cannot exceed CCUS_CONDITIONAL_LEVELS")
    all(r -> (CONDITIONAL_LEVELS - 1) % (r - 1) == 0, CONDITIONAL_RESOLUTIONS) ||
        error("Every resolution must be embedded in CCUS_CONDITIONAL_LEVELS")

    mkpath(RESULT_DIR)
    p = pareto_parameters()
    anchors = anchor_data(p)
    frontier = CSV.read(result_path("pareto_frontier.csv"), DataFrame)
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))
    lower = [minimum(Float64.(frontier[!, name])) for name in (:TAC, :TotEmiss, :ISI)]
    upper = [maximum(Float64.(frontier[!, name])) for name in (:TAC, :TotEmiss, :ISI)]
    ranges = upper .- lower
    scaled_frontier = hcat([
        (Float64.(frontier[!, name]) .- lower[index]) ./ ranges[index]
        for (index, name) in enumerate((:TAC, :TotEmiss, :ISI))
    ]...)

    specs = [
        (grouping="TAC + total emissions", grouped=(:TAC, :TotEmiss), retained=:ISI),
        (grouping="TAC + ISI", grouped=(:TAC, :ISI), retained=:TotEmiss),
        (grouping="Total emissions + ISI", grouped=(:TotEmiss, :ISI), retained=:TAC),
    ]

    endpoint_rows = NamedTuple[]
    slice_rows = NamedTuple[]
    solve_log = NamedTuple[]
    fractions = collect(range(0.0, 1.0, length=CONDITIONAL_LEVELS))

    for spec in specs
        retained_index = objective_index(spec.retained)
        warm = Dict{Symbol, Union{Nothing, Vector{Float64}}}(
            spec.grouped[1] => nothing,
            spec.grouped[2] => nothing,
        )
        for (level_index, fraction) in enumerate(fractions)
            target = lower[retained_index] + fraction * ranges[retained_index]
            endpoints = Dict{Symbol, Any}()
            for (primary, secondary) in (spec.grouped, reverse(spec.grouped))
                nearest = nearest_frontier_starts(frontier, z_names, spec.retained, target)
                starts = distinct_starts(vcat(
                    isnothing(warm[primary]) ? Vector{Vector{Float64}}() : [warm[primary]],
                    nearest,
                    [anchor.z for anchor in anchors],
                ))
                outcome = solve_lexicographic_endpoint(
                    p,
                    lower,
                    ranges,
                    starts;
                    retained=spec.retained,
                    target,
                    primary,
                    secondary,
                )
                accepted = !isnothing(outcome.best)
                push!(solve_log, (
                    grouping=spec.grouping,
                    retained=String(spec.retained),
                    level_index,
                    level_fraction=fraction,
                    retained_target=target,
                    endpoint="min_$(primary)",
                    accepted,
                    stage_one_attempts=outcome.stage_one_attempts,
                    stage_one_successes=outcome.stage_one_successes,
                    stage_two_attempts=outcome.stage_two_attempts,
                    stage_two_successes=outcome.stage_two_successes,
                    lexicographic_completed=outcome.lexicographic_completed,
                ))
                if accepted
                    result = outcome.best
                    warm[primary] = result.z
                    candidate = normalized_objectives(result.objectives, lower, ranges)
                    sample_dominates = any(
                        all(scaled_frontier[row, :] .<= candidate .+ 1.0e-6) &&
                        any(scaled_frontier[row, :] .< candidate .- 1.0e-6)
                        for row in axes(scaled_frontier, 1)
                    )
                    push!(endpoint_rows, endpoint_row(
                        spec.grouping,
                        spec.retained,
                        level_index,
                        fraction,
                        target,
                        primary,
                        secondary,
                        outcome,
                        sample_dominates,
                        z_names,
                    ))
                    endpoints[primary] = outcome
                end
            end

            if all(haskey(endpoints, objective) for objective in spec.grouped)
                first, second = spec.grouped
                first_values = [
                    objective_value(endpoints[first].best.objectives, first),
                    objective_value(endpoints[second].best.objectives, first),
                ]
                second_values = [
                    objective_value(endpoints[first].best.objectives, second),
                    objective_value(endpoints[second].best.objectives, second),
                ]
                first_range = (maximum(first_values) - minimum(first_values)) /
                              ranges[objective_index(first)]
                second_range = (maximum(second_values) - minimum(second_values)) /
                               ranges[objective_index(second)]
                push!(slice_rows, (
                    grouping=spec.grouping,
                    retained=String(spec.retained),
                    level_index,
                    level_fraction=fraction,
                    retained_target=target,
                    grouped_objective_1=String(first),
                    grouped_objective_2=String(second),
                    normalized_range_1=first_range,
                    normalized_range_2=second_range,
                    information_loss=first_range + second_range,
                    both_lexicographic=endpoints[first].lexicographic_completed &&
                                       endpoints[second].lexicographic_completed,
                ))
            end
            println(spec.grouping, " | exact slice ", level_index, "/", CONDITIONAL_LEVELS,
                    " | endpoints=", length(endpoints), "/2")
        end
    end

    endpoints = DataFrame(endpoint_rows)
    slices = DataFrame(slice_rows)
    logs = DataFrame(solve_log)
    summary_rows = NamedTuple[]
    for resolution in CONDITIONAL_RESOLUTIONS
        stride = (CONDITIONAL_LEVELS - 1) ÷ (resolution - 1)
        selected = collect(1:stride:CONDITIONAL_LEVELS)
        for spec in specs
            subset = slices[
                (slices.grouping .== spec.grouping) .& in.(slices.level_index, Ref(selected)),
                :,
            ]
            push!(summary_rows, (
                resolution,
                grouping=spec.grouping,
                requested_slices=resolution,
                feasible_slices=nrow(subset),
                mean_information_loss=mean(subset.information_loss),
                sd_information_loss=nrow(subset) > 1 ? std(subset.information_loss) : 0.0,
                min_information_loss=minimum(subset.information_loss),
                max_information_loss=maximum(subset.information_loss),
            ))
        end
    end
    summary = DataFrame(summary_rows)
    transform!(
        groupby(summary, :resolution),
        :mean_information_loss => (x -> invperm(sortperm(x))) => :rank,
    )

    CSV.write(conditional_path("endpoints.csv"), endpoints)
    CSV.write(conditional_path("slices.csv"), slices)
    CSV.write(conditional_path("summary.csv"), summary)
    CSV.write(conditional_path("solve_log.csv"), logs)
    open(conditional_path("run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "profile = ", PARETO_PROFILE)
        println(io, "method = Russell-Allman equations 10-11 with exact retained-objective equalities")
        println(io, "conditional_levels = ", CONDITIONAL_LEVELS)
        println(io, "embedded_resolutions = ", CONDITIONAL_RESOLUTIONS)
        println(io, "lexicographic_tolerance_normalized = ", LEXICOGRAPHIC_TOLERANCE)
        println(io, "retained_acceptance_tolerance_normalized = ", RETAINED_TOLERANCE)
        println(io, "starts_per_target = ", STARTS_PER_TARGET)
        println(io, "global_optimality = not claimed; deterministic Ipopt multi-start local NLP solves")
        println(io, "objective_lower = ", lower)
        println(io, "objective_upper = ", upper)
        println(io, "accepted_endpoints = ", nrow(endpoints), "/", 2 * length(specs) * CONDITIONAL_LEVELS)
        println(io, "complete_slices = ", nrow(slices), "/", length(specs) * CONDITIONAL_LEVELS)
        println(io, "lexicographic_fallbacks = ", count(.!endpoints.lexicographic_completed))
        println(io, "sample_dominated_endpoints = ", count(endpoints.dominated_by_sample))
        println(io, "max_retained_normalized_error = ", maximum(endpoints.retained_normalized_error))
        println(io, "max_abs_equality = ", maximum(endpoints.max_abs_equality))
        println(io, "max_inequality_violation = ", maximum(endpoints.max_inequality_violation))
    end

    println(summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
