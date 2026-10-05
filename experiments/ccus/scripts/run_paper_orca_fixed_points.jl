using CSV
using DataFrames
using Dates
using ForwardDiff
using LinearAlgebra
using Random
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const CONVERGED = (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
const PRIMAL_OK = (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
const FEASIBILITY_TOLERANCE = 1.0e-6
const OBJECTIVES = (:TAC, :TotEmiss, :ISI)

function accepted(result)
    result.z === nothing && return false
    result.termination in CONVERGED || return false
    result.primal_status in PRIMAL_OK || return false
    r = result.residuals
    return r.max_abs_equality <= FEASIBILITY_TOLERANCE &&
           r.max_abs_source_carbon_diagnostic <= FEASIBILITY_TOLERANCE &&
           r.max_inequality_violation <= FEASIBILITY_TOLERANCE
end

function distinct_starts(starts)
    output = Vector{Vector{Float64}}()
    for start in starts
        start === nothing && continue
        if all(maximum(abs.(start .- existing)) > 1.0e-8 for existing in output)
            push!(output, Float64.(start))
        end
    end
    return output
end

function solve_anchor_points(p)
    feasibility = solve_ccus(p; objective=:feasibility, time_limit=60.0)
    accepted(feasibility) || error("Feasibility seed failed: $(feasibility.termination)")

    preliminary = Dict{Symbol, CCUSSolveResult}()
    for objective in OBJECTIVES
        result = solve_ccus(p; objective, start=feasibility.z, time_limit=60.0)
        accepted(result) || error("Preliminary $objective seed failed: $(result.termination)")
        preliminary[objective] = result
    end

    starts = distinct_starts(vcat([feasibility.z], [preliminary[o].z for o in OBJECTIVES]))
    anchors = Dict{Symbol, CCUSSolveResult}()
    for objective in OBJECTIVES
        candidates = CCUSSolveResult[]
        for start in starts
            result = solve_ccus(p; objective, start, time_limit=60.0)
            accepted(result) && push!(candidates, result)
        end
        isempty(candidates) && error("No accepted anchor for $objective")
        anchors[objective] = candidates[argmin(
            getproperty(candidate.objective_values, objective) for candidate in candidates
        )]
    end
    return anchors
end

function safe_normalize(vector; tolerance=1.0e-14)
    magnitude = norm(vector)
    magnitude <= tolerance && return zeros(length(vector))
    return Float64.(vector) ./ magnitude
end

function conic_direction(z, p, rng)
    objective_jacobian = ForwardDiff.jacobian(x -> objective_vector(x, p), z)
    inequality_jacobian = ForwardDiff.jacobian(x -> inequality_values(x, p), z)
    inequality_at_z = inequality_values(z, p)
    constraint_index = argmax(inequality_at_z)
    normal = inequality_jacobian[constraint_index, :]
    denominator = dot(normal, normal)

    projected = zeros(length(OBJECTIVES), length(z))
    for objective_index in eachindex(OBJECTIVES)
        descent = safe_normalize(-objective_jacobian[objective_index, :])
        candidate = denominator <= 1.0e-14 ? descent :
                    descent .- dot(descent, normal) / denominator .* normal
        projected[objective_index, :] .= safe_normalize(candidate)
        norm(projected[objective_index, :]) <= 1.0e-14 &&
            (projected[objective_index, :] .= descent)
    end

    weights = rand(rng, length(OBJECTIVES))
    delta = vec(sum(projected .* weights, dims=1) ./ sum(weights))
    return safe_normalize(delta), constraint_index, inequality_at_z[constraint_index]
end

function project_point(p, target; time_limit=30.0)
    return solve_ccus(p; objective=:feasibility, start=target, time_limit)
end

function z_tuple(z)
    names = Tuple(Symbol.("z" .* string.(eachindex(z))))
    return NamedTuple{names}(Tuple(z))
end

function g_tuple(values)
    names = Tuple(Symbol.("g" .* string.(eachindex(values))))
    return NamedTuple{names}(Tuple(values))
end

function gradient_tuple(values)
    names = Tuple(Symbol.("dz" .* string.(eachindex(values))))
    return NamedTuple{names}(Tuple(values))
end

function point_row(label, replicate, rng_seed, step_size, points_per_seed,
                   seed_objective, sequence, point_id, is_seed, z, p;
                   selected_constraint=missing, selected_constraint_value=missing,
                   projection_distance=0.0)
    o = objective_values(z, p)
    r = residual_summary(z, p)
    return merge((
        run_label=label,
        replicate,
        rng_seed,
        step_size,
        points_per_seed,
        seed_objective=String(seed_objective),
        sequence,
        point_id,
        is_seed,
        selected_constraint,
        selected_constraint_value,
        projection_distance,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        max_abs_equality=r.max_abs_equality,
        max_abs_source_carbon_diagnostic=r.max_abs_source_carbon_diagnostic,
        max_inequality_violation=r.max_inequality_violation,
    ), z_tuple(z), g_tuple(inequality_values(z, p)))
end

function profile_parameters(profile)
    profile == "direct_use_expansion" && return direct_use_expansion_parameters()
    profile == "supply_chain_condition1" && return supply_chain_condition1_parameters()
    profile == "supply_chain_condition2" && return supply_chain_condition2_parameters()
    error("Unknown CCUS_ORCA_PROFILE: $profile")
end

function main()
    mkpath(RESULT_DIR)
    label = get(ENV, "CCUS_ORCA_RUN_LABEL", "paper_main")
    step_size = parse(Float64, get(ENV, "CCUS_ORCA_STEP_SIZE", "0.03"))
    points_per_seed = parse(Int, get(ENV, "CCUS_ORCA_POINTS_PER_SEED", "40"))
    replicates = parse(Int, get(ENV, "CCUS_ORCA_REPLICATES", "20"))
    first_seed = parse(Int, get(ENV, "CCUS_ORCA_FIRST_RANDOM_SEED", "20250810"))
    min_point_distance = parse(Float64, get(ENV, "CCUS_ORCA_MIN_POINT_DISTANCE", "1e-8"))
    max_projection_failures = parse(Int, get(ENV, "CCUS_ORCA_MAX_PROJECTION_FAILURES", "20"))
    profile = get(ENV, "CCUS_ORCA_PROFILE", "direct_use_expansion")
    step_size > 0 || error("CCUS_ORCA_STEP_SIZE must be positive")
    points_per_seed >= 1 || error("CCUS_ORCA_POINTS_PER_SEED must be positive")
    replicates >= 1 || error("CCUS_ORCA_REPLICATES must be positive")

    p = profile_parameters(profile)
    anchors = solve_anchor_points(p)
    point_rows = NamedTuple[]
    attempt_rows = NamedTuple[]
    gradient_rows = NamedTuple[]

    for replicate in 1:replicates
        rng_seed = first_seed + replicate - 1
        rng = MersenneTwister(rng_seed)
        selected = Vector{Vector{Float64}}()

        for seed_objective in OBJECTIVES
            initial_projection = project_point(p, anchors[seed_objective].z)
            accepted(initial_projection) || error(
                "Initial projection failed for replicate $replicate, $seed_objective"
            )
            current = initial_projection.z
            seed_point_id = "R$(lpad(replicate, 2, '0'))_$(seed_objective)_000"
            if all(norm(current .- existing) >= min_point_distance for existing in selected)
                push!(selected, current)
            end
            push!(point_rows, point_row(
                label, replicate, rng_seed, step_size, points_per_seed,
                seed_objective, 0, seed_point_id, true, current, p,
            ))

            accepted_from_seed = 0
            failures = 0
            attempt = 0
            while accepted_from_seed < points_per_seed && failures < max_projection_failures
                attempt += 1
                delta, constraint_index, constraint_value = conic_direction(current, p, rng)
                trial = current .+ step_size .* delta
                projection = project_point(p, trial)
                projection_ok = accepted(projection)
                kept = false
                distance = missing
                if projection_ok
                    candidate = projection.z
                    distance = norm(candidate .- trial)
                    kept = all(norm(candidate .- existing) >= min_point_distance for existing in selected)
                    current = candidate
                    if kept
                        accepted_from_seed += 1
                        failures = 0
                        push!(selected, candidate)
                        point_id = "R$(lpad(replicate, 2, '0'))_$(seed_objective)_$(lpad(accepted_from_seed, 3, '0'))"
                        push!(point_rows, point_row(
                            label, replicate, rng_seed, step_size, points_per_seed,
                            seed_objective, accepted_from_seed, point_id, false,
                            candidate, p;
                            selected_constraint=constraint_index,
                            selected_constraint_value=constraint_value,
                            projection_distance=distance,
                        ))
                    else
                        failures += 1
                    end
                else
                    failures += 1
                end
                push!(attempt_rows, (
                    run_label=label,
                    replicate,
                    rng_seed,
                    seed_objective=String(seed_objective),
                    attempt,
                    accepted_from_seed,
                    selected_constraint=constraint_index,
                    selected_constraint_name=inequality_names()[constraint_index],
                    selected_constraint_value=constraint_value,
                    projection_accepted=projection_ok,
                    kept,
                    projection_distance=distance,
                    termination=string(projection.termination),
                    primal_status=string(projection.primal_status),
                    solve_time_s=projection.solve_time,
                ))
            end
            println("replicate=", replicate, "/", replicates,
                    " seed=", seed_objective,
                    " selected=", accepted_from_seed, "/", points_per_seed,
                    " attempts=", attempt)
        end
    end

    points = DataFrame(point_rows)
    for row in eachrow(points)
        z = Float64[row[Symbol("z$i")] for i in 1:length(variable_names())]
        jacobian = ForwardDiff.jacobian(x -> objective_vector(x, p), z)
        for (objective_index, objective) in enumerate(OBJECTIVES)
            push!(gradient_rows, merge((
                run_label=label,
                replicate=row.replicate,
                point_id=row.point_id,
                objective=String(objective),
            ), gradient_tuple(jacobian[objective_index, :])))
        end
    end

    reference = Float64[points[1, Symbol("z$i")] for i in 1:length(variable_names())]
    equality_jacobian = ForwardDiff.jacobian(x -> equality_values(x, p), reference)
    inequality_jacobian = ForwardDiff.jacobian(x -> inequality_values(x, p), reference)
    equality_rows = [merge((constraint_index=i, constraint=equality_names()[i]),
                           gradient_tuple(equality_jacobian[i, :]))
                     for i in axes(equality_jacobian, 1)]
    inequality_rows = [merge((constraint_index=i, constraint=inequality_names()[i]),
                             gradient_tuple(inequality_jacobian[i, :]))
                       for i in axes(inequality_jacobian, 1)]

    prefix = "$(profile)_paper_orca_$(label)_"
    CSV.write(joinpath(RESULT_DIR, prefix * "points.csv"), points)
    CSV.write(joinpath(RESULT_DIR, prefix * "projection_attempts.csv"), DataFrame(attempt_rows))
    CSV.write(joinpath(RESULT_DIR, prefix * "objective_gradients.csv"), DataFrame(gradient_rows))
    CSV.write(joinpath(RESULT_DIR, prefix * "equality_jacobian.csv"), DataFrame(equality_rows))
    CSV.write(joinpath(RESULT_DIR, prefix * "inequality_jacobian.csv"), DataFrame(inequality_rows))

    open(joinpath(RESULT_DIR, prefix * "generation_run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "profile = ", profile)
        println(io, "method = Wang-Allman 2025 equations 3-4 mirrored in Julia with Ipopt projection")
        println(io, "run_label = ", label)
        println(io, "step_size = ", step_size)
        println(io, "points_per_seed_additional = ", points_per_seed)
        println(io, "replicates = ", replicates)
        println(io, "first_random_seed = ", first_seed)
        println(io, "selected_points = ", nrow(points))
        println(io, "projection_attempts = ", length(attempt_rows))
        println(io, "projection_failures = ", count(row -> !row.projection_accepted, attempt_rows))
        println(io, "duplicate_rejections = ", count(row -> row.projection_accepted && !row.kept, attempt_rows))
        println(io, "max_abs_equality = ", maximum(points.max_abs_equality))
        println(io, "max_inequality_violation = ", maximum(points.max_inequality_violation))
    end
end

main()
