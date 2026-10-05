using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "run_pareto_epsilon.jl"))

const EXACT_LABEL = get(ENV, "CCUS_EXACT_INFO_LABEL", "main")
const RETAINED_LEVELS = parse(Int, get(ENV, "CCUS_EXACT_RETAINED_LEVELS", "21"))
const EPSILON_LEVELS = parse(Int, get(ENV, "CCUS_EXACT_EPSILON_LEVELS", "21"))
const ACTIVE_TOLERANCE = 1.0e-6
const EXACT_SOLVE_TIME_LIMIT = parse(
    Float64, get(ENV, "CCUS_EXACT_SOLVE_TIME_LIMIT", "5.0")
)
const REUSE_LABEL = get(ENV, "CCUS_EXACT_REUSE_LABEL", "")

exact_path(name) = result_path("exact_info_loss_$(EXACT_LABEL)_$(name)")
objective_index(name::Symbol) = name == :TAC ? 1 : name == :TotEmiss ? 2 : 3
objective_value(o, name::Symbol) = getproperty(o, name)

function nearest_starts(frontier, z_names, lower, ranges, retained, retained_target,
                        swept, swept_target; count=3)
    retained_index = objective_index(retained)
    swept_index = objective_index(swept)
    distances = abs.((Float64.(frontier[!, retained]) .- retained_target) ./ ranges[retained_index]) .+
                abs.((Float64.(frontier[!, swept]) .- swept_target) ./ ranges[swept_index])
    indices = sortperm(distances)[1:min(count, nrow(frontier))]
    return [Float64.(collect(frontier[index, z_names])) for index in indices]
end

function mode_arguments(mode, cap)
    if mode == :full_TAC
        return cap[:TotEmiss], cap[:ISI]
    elseif mode == :full_TotEmiss
        return cap[:TAC], cap[:ISI]
    elseif mode == :full_ISI
        return cap[:TAC], cap[:TotEmiss]
    end
    error("Unknown scalarization mode: $mode")
end

function solve_exact_starts(
    p,
    lower,
    ranges,
    anchors;
    mode,
    epsilon_a,
    epsilon_b,
    warm_starts,
)
    ordered_anchors = sort(
        anchors,
        by=anchor -> (
            target_violation(anchor.objectives, ranges, mode, epsilon_a, epsilon_b),
            sum(abs, normalized_objectives(anchor.objectives, lower, ranges)),
        ),
    )
    starts = distinct_starts(vcat(warm_starts, [anchor.z for anchor in ordered_anchors]))
    candidates = NamedTuple[]
    attempts = 0
    first_batch = min(3, length(starts))
    for start in starts[1:first_batch]
        attempts += 1
        result = solve_scalarization(
            p,
            lower,
            ranges;
            mode,
            epsilon_a,
            epsilon_b,
            start,
            time_limit=EXACT_SOLVE_TIME_LIMIT,
        )
        result.accepted && push!(candidates, result)
    end
    if isempty(candidates)
        last_attempt = min(first_batch + 2, length(starts))
        for start in starts[(first_batch + 1):last_attempt]
            attempts += 1
            result = solve_scalarization(
                p,
                lower,
                ranges;
                mode,
                epsilon_a,
                epsilon_b,
                start,
                time_limit=EXACT_SOLVE_TIME_LIMIT,
            )
            result.accepted && push!(candidates, result)
        end
    end
    isempty(candidates) && return (; best=nothing, attempts, successful=0)
    best = candidates[argmin(getfield.(candidates, :scalarized))]
    return (; best, attempts, successful=length(candidates))
end

function point_row(spec, orientation, retained_index, retained_fraction,
                   retained_target, epsilon_index, epsilon_fraction,
                   swept_target, swept_min_bound, result, attempts, successes,
                   z_names, lower, ranges)
    o = result.objectives
    r = result.residuals
    retained_error = abs(objective_value(o, orientation.retained) - retained_target) /
                     ranges[objective_index(orientation.retained)]
    swept_slack = (swept_target - objective_value(o, orientation.swept)) /
                  ranges[objective_index(orientation.swept)]
    z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
    return merge((
        grouping=spec.grouping,
        orientation=orientation.name,
        mode=String(orientation.mode),
        primary=String(orientation.primary),
        swept=String(orientation.swept),
        retained=String(orientation.retained),
        retained_index,
        retained_fraction,
        retained_target,
        epsilon_index,
        epsilon_fraction,
        swept_target,
        swept_min_bound,
        retained_active=retained_error <= ACTIVE_TOLERANCE,
        retained_normalized_error=retained_error,
        swept_normalized_slack=swept_slack,
        attempts,
        successful_starts=successes,
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
        TAC_normalized=(o.TAC - lower[1]) / ranges[1],
        TotEmiss_normalized=(o.TotEmiss - lower[2]) / ranges[2],
        ISI_normalized=(o.ISI - lower[3]) / ranges[3],
    ), z_tuple)
end

function main()
    RETAINED_LEVELS >= 3 || error("CCUS_EXACT_RETAINED_LEVELS must be at least 3")
    EPSILON_LEVELS >= 3 || error("CCUS_EXACT_EPSILON_LEVELS must be at least 3")
    mkpath(RESULT_DIR)

    p = pareto_parameters()
    anchors = anchor_data(p)
    frontier = CSV.read(result_path("pareto_frontier.csv"), DataFrame)
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))
    objective_symbols = (:TAC, :TotEmiss, :ISI)
    lower = [minimum(Float64.(frontier[!, name])) for name in objective_symbols]
    upper = [maximum(Float64.(frontier[!, name])) for name in objective_symbols]
    ranges = upper .- lower

    specs = [
        (
            grouping="TAC + total emissions",
            orientations=(
                (name="TAC_primary", mode=:full_TAC, primary=:TAC,
                 swept=:TotEmiss, retained=:ISI),
            ),
        ),
        (
            grouping="TAC + ISI",
            orientations=(
                (name="TAC_primary", mode=:full_TAC, primary=:TAC,
                 swept=:ISI, retained=:TotEmiss),
            ),
        ),
        (
            grouping="Total emissions + ISI",
            orientations=(
                (name="TotEmiss_primary", mode=:full_TotEmiss, primary=:TotEmiss,
                 swept=:ISI, retained=:TAC),
            ),
        ),
    ]

    epsilon_fractions = collect(range(1.0, 0.0, length=EPSILON_LEVELS))
    rows = NamedTuple[]
    logs = NamedTuple[]
    reused_targets = 0
    reuse_points = nothing
    reuse_logs = nothing
    reuse_epsilon_levels = 0
    reuse_stride = 0
    if !isempty(REUSE_LABEL)
        reuse_points = CSV.read(
            result_path("exact_info_loss_$(REUSE_LABEL)_points.csv"), DataFrame
        )
        reuse_logs = CSV.read(
            result_path("exact_info_loss_$(REUSE_LABEL)_solve_log.csv"), DataFrame
        )
        reuse_epsilon_levels = maximum(Int.(reuse_logs.epsilon_index))
        (EPSILON_LEVELS - 1) % (reuse_epsilon_levels - 1) == 0 || error(
            "Reused epsilon grid must be embedded in the requested grid"
        )
        reuse_stride = (EPSILON_LEVELS - 1) ÷ (reuse_epsilon_levels - 1)
    end

    for spec in specs
        for orientation in spec.orientations
            retained_objective_index = objective_index(orientation.retained)
            swept_objective_index = objective_index(orientation.swept)
            available_targets = sort(unique(Float64.(frontier[!, orientation.retained])))
            target_indices = round.(
                Int, range(1, length(available_targets), length=RETAINED_LEVELS)
            )
            retained_targets = available_targets[target_indices]
            above = Union{Nothing, Vector{Float64}}[nothing for _ in 1:EPSILON_LEVELS]
            for (retained_index, retained_target) in enumerate(retained_targets)
                retained_fraction = (retained_target - lower[retained_objective_index]) /
                                    ranges[retained_objective_index]
                witness_ids = findall(
                    Float64.(frontier[!, orientation.retained]) .<= retained_target +
                    FEASIBILITY_TOLERANCE * ranges[retained_objective_index]
                )
                isempty(witness_ids) && error(
                    "No full-frontier witness satisfies $(orientation.retained) <= $retained_target"
                )
                witness_index = witness_ids[argmin(
                    Float64.(frontier[witness_ids, orientation.swept])
                )]
                swept_min_bound = Float64(frontier[witness_index, orientation.swept])
                left = nothing
                current = Union{Nothing, Vector{Float64}}[
                    nothing for _ in 1:EPSILON_LEVELS
                ]
                accepted_in_slice = 0
                active_in_slice = 0
                for (epsilon_index, epsilon_fraction) in enumerate(epsilon_fractions)
                    swept_target = swept_min_bound + epsilon_fraction *
                                   (upper[swept_objective_index] - swept_min_bound)
                    if !isnothing(reuse_logs) && (epsilon_index - 1) % reuse_stride == 0
                        old_epsilon_index = (epsilon_index - 1) ÷ reuse_stride + 1
                        old_log = reuse_logs[
                            (reuse_logs.grouping .== spec.grouping) .&
                            (reuse_logs.orientation .== orientation.name) .&
                            (reuse_logs.retained_index .== retained_index) .&
                            (reuse_logs.epsilon_index .== old_epsilon_index),
                            :,
                        ]
                        if nrow(old_log) == 1 && Bool(old_log.accepted[1])
                            old_point = reuse_points[
                                (reuse_points.grouping .== spec.grouping) .&
                                (reuse_points.orientation .== orientation.name) .&
                                (reuse_points.retained_index .== retained_index) .&
                                (reuse_points.epsilon_index .== old_epsilon_index),
                                :,
                            ]
                            if nrow(old_point) == 1
                                point = old_point[1, :]
                                active = Bool(point.retained_active)
                                push!(rows, merge(
                                    NamedTuple(point),
                                    (; epsilon_index, epsilon_fraction, swept_target),
                                ))
                                push!(logs, merge(
                                    NamedTuple(old_log[1, :]),
                                    (; epsilon_index, epsilon_fraction, swept_target),
                                ))
                                z = Float64.(collect(point[z_names]))
                                left = z
                                current[epsilon_index] = z
                                accepted_in_slice += 1
                                active_in_slice += active
                                reused_targets += 1
                                continue
                            end
                        end
                    end
                    nearest = nearest_starts(
                        frontier,
                        z_names,
                        lower,
                        ranges,
                        orientation.retained,
                        retained_target,
                        orientation.swept,
                        swept_target,
                    )
                    warm_starts = distinct_starts(vcat(
                        isnothing(left) ? Vector{Vector{Float64}}() : [left],
                        isnothing(above[epsilon_index]) ? Vector{Vector{Float64}}() :
                                                        [above[epsilon_index]],
                        [Float64.(collect(frontier[witness_index, z_names]))],
                        nearest,
                    ))
                    epsilon_a, epsilon_b = mode_arguments(
                        orientation.mode,
                        Dict(
                            orientation.retained => retained_target,
                            orientation.swept => swept_target,
                        ),
                    )
                    outcome = solve_exact_starts(
                        p,
                        lower,
                        ranges,
                        anchors;
                        mode=orientation.mode,
                        epsilon_a,
                        epsilon_b,
                        warm_starts,
                    )
                    accepted = !isnothing(outcome.best)
                    active = false
                    if accepted
                        result = outcome.best
                        accepted_in_slice += 1
                        active = abs(
                            objective_value(result.objectives, orientation.retained) -
                            retained_target,
                        ) / ranges[retained_objective_index] <= ACTIVE_TOLERANCE
                        active_in_slice += active
                        push!(rows, point_row(
                            spec,
                            orientation,
                            retained_index,
                            retained_fraction,
                            retained_target,
                            epsilon_index,
                            epsilon_fraction,
                            swept_target,
                            swept_min_bound,
                            result,
                            outcome.attempts,
                            outcome.successful,
                            z_names,
                            lower,
                            ranges,
                        ))
                        left = result.z
                        current[epsilon_index] = result.z
                    end
                    push!(logs, (
                        grouping=spec.grouping,
                        orientation=orientation.name,
                        retained=String(orientation.retained),
                        retained_index,
                        retained_fraction,
                        retained_target,
                        swept=String(orientation.swept),
                        epsilon_index,
                        epsilon_fraction,
                        swept_target,
                        attempts=outcome.attempts,
                        successful_starts=outcome.successful,
                        accepted,
                        retained_active=active,
                    ))
                end
                above = current
                println(spec.grouping, " / ", orientation.name,
                        " | retained level ", retained_index, "/", RETAINED_LEVELS,
                        " | accepted=", accepted_in_slice, "/", EPSILON_LEVELS,
                        " | active=", active_in_slice)
            end
        end
    end

    points = DataFrame(rows)
    solve_log = DataFrame(logs)
    CSV.write(exact_path("points.csv"), points)
    CSV.write(exact_path("solve_log.csv"), solve_log)
    open(exact_path("run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "profile = ", PARETO_PROFILE)
        println(io, "method = exact retained-objective active slices of augmented epsilon-constraint Pareto solves")
        println(io, "retained_levels = ", RETAINED_LEVELS)
        println(io, "epsilon_levels = ", EPSILON_LEVELS)
        println(io, "orientations_per_grouping = 1; epsilon-constraint is nonconvex-front capable")
        println(io, "active_tolerance_normalized = ", ACTIVE_TOLERANCE)
        println(io, "per_solve_time_limit_s = ", EXACT_SOLVE_TIME_LIMIT)
        println(io, "deterministic_starts = 3 initial; 2 additional only if none succeeds")
        println(io, "reuse_label = ", isempty(REUSE_LABEL) ? "none" : REUSE_LABEL)
        println(io, "reused_accepted_targets = ", reused_targets)
        println(io, "global_optimality = not claimed; deterministic Ipopt multi-start local NLP solves")
        println(io, "objective_lower_initial = ", lower)
        println(io, "objective_upper_initial = ", upper)
        println(io, "swept_lower_bound = minimum verified full-frontier witness satisfying retained cap")
        println(io, "retained_targets = quantiles of actual retained-objective values on verified frontier")
        println(io, "target_solves = ", length(specs) * RETAINED_LEVELS * EPSILON_LEVELS)
        println(io, "accepted_solves = ", nrow(points))
        println(io, "active_slice_points = ", count(points.retained_active))
        println(io, "max_abs_equality = ", maximum(points.max_abs_equality))
        println(io, "max_inequality_violation = ", maximum(points.max_inequality_violation))
        println(io, "max_epsilon_violation = ", maximum(points.epsilon_violation))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
