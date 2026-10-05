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
const OBJECTIVES = (:TAC, :TotEmiss, :ISI)
const FEASIBILITY_TOLERANCE = 1.0e-6

function accepted(result)
    result.z !== nothing || return false
    result.termination in CONVERGED || return false
    result.primal_status in PRIMAL_OK || return false
    r = result.residuals
    return r.max_abs_equality <= FEASIBILITY_TOLERANCE &&
           r.max_abs_source_carbon_diagnostic <= FEASIBILITY_TOLERANCE &&
           r.max_inequality_violation <= FEASIBILITY_TOLERANCE
end

function scale_sink_columns(matrix, factors)
    return matrix .* reshape(Float64.(factors), 1, :)
end

function parameter_profiles()
    base = baseline_parameters()
    unforced = with_parameters(base; min_sink1_flow=0.0)

    three_way = with_parameters(
        unforced;
        Gmax_k=Float64[1000, 650, 1500, 756, 900, 913],
        Mu_k=Float64[0.24, 0.40, 0.00, 0.18, 0.12, 0.30],
        sink_safety=Float64[4, 8, 18, 32, 25, 30],
        compressor_power=scale_sink_columns(base.compressor_power, [1.0, 1.0, 1.0, 1.0, 0.18, 1.0]),
        pump_power=scale_sink_columns(base.pump_power, [1.0, 1.0, 1.0, 1.0, 0.18, 1.0]),
        distance=scale_sink_columns(base.distance, [1.0, 1.0, 3.0, 1.5, 0.30, 1.5]),
    )

    direct_use = direct_use_expansion_parameters()

    limited_hubs = with_parameters(
        unforced;
        Zmin_k=Float64[0.06, 0.30, 0.95, 0.999, 0.999, 0.999],
        Gmax_k=Float64[220, 500, 360, 500, 360, 500],
        Mu_k=Float64[0.24, 0.16, 0.00, 0.14, 0.12, 0.22],
        sink_safety=Float64[5, 4, 16, 28, 22, 26],
        compressor_power=scale_sink_columns(base.compressor_power, [1.0, 1.0, 1.0, 0.8, 0.18, 0.9]),
        pump_power=scale_sink_columns(base.pump_power, [1.0, 1.0, 1.0, 0.8, 0.18, 0.9]),
        distance=scale_sink_columns(base.distance, [1.0, 0.35, 3.0, 1.2, 0.30, 1.2]),
    )

    expanded_greenhouse = with_parameters(
        unforced;
        Gmax_k=Float64[250, 1800, 1300, 756, 800, 913],
        Mu_k=Float64[0.24, 0.12, 0.00, 0.18, 0.12, 0.30],
        sink_safety=Float64[7, 3, 18, 32, 25, 30],
        compressor_power=scale_sink_columns(base.compressor_power, [1.0, 1.0, 1.0, 1.0, 0.18, 1.0]),
        pump_power=scale_sink_columns(base.pump_power, [1.0, 1.0, 1.0, 1.0, 0.18, 1.0]),
        distance=scale_sink_columns(base.distance, [1.0, 0.5, 3.0, 1.5, 0.30, 1.5]),
    )

    zero_carbon_power = with_parameters(
        three_way;
        power_carbon_intensity=0.05,
        treatment_emission_factor=0.015,
    )

    high_capture = with_parameters(
        limited_hubs;
        capture_fraction=0.60,
        Gmax_k=Float64[300, 700, 500, 650, 450, 650],
    )

    return [
        (name="baseline_forced", evidence="legacy baseline", rationale="Original parameters and forced Algae flow.", p=base),
        (name="baseline_unforced", evidence="diagnostic", rationale="Removes the artificial minimum Algae-flow constraint only.", p=unforced),
        (name="three_way_competition", evidence="exploratory", rationale="Low-cost Urea transport, zero-loss saline storage, and expanded low-risk Algae capacity.", p=three_way),
        (name="direct_use_expansion", evidence="exploratory", rationale="Adds a large lower-purity Greenhouse route to expose preprocessing choices.", p=direct_use),
        (name="capacity_limited_hubs", evidence="exploratory", rationale="Limits preferred hubs so optimal systems must combine multiple sinks.", p=limited_hubs),
        (name="expanded_greenhouse", evidence="exploratory", rationale="Expands a low-safety-index Greenhouse market while retaining its high purity requirement.", p=expanded_greenhouse),
        (name="low_carbon_energy", evidence="exploratory", rationale="Tests whether cleaner power weakens the preprocessing and transport emission penalty.", p=zero_carbon_power),
        (name="high_capture_limited", evidence="exploratory", rationale="Raises capture to 60% under finite sink capacities to create network-wide allocation pressure.", p=high_capture),
    ]
end

function distinct_starts(starts)
    unique_starts = Vector{Vector{Float64}}()
    for start in starts
        start === nothing && continue
        if all(maximum(abs.(start .- existing)) > 1.0e-8 for existing in unique_starts)
            push!(unique_starts, Float64.(start))
        end
    end
    return unique_starts
end

objective_value(result, objective) = getproperty(result.objective_values, objective)

function best_from_starts(p, objective, starts)
    candidates = CCUSSolveResult[]
    for start in distinct_starts(starts)
        result = solve_ccus(p; objective, start, time_limit=45.0)
        accepted(result) && push!(candidates, result)
    end
    isempty(candidates) && return nothing
    return candidates[argmin(objective_value(result, objective) for result in candidates)]
end

function solve_profile(p)
    feasibility = solve_ccus(p; objective=:feasibility, time_limit=60.0)
    accepted(feasibility) || return (feasibility=feasibility, solutions=Dict{Symbol, CCUSSolveResult}())

    anchors = Dict{Symbol, CCUSSolveResult}()
    for objective in OBJECTIVES
        result = best_from_starts(p, objective, [feasibility.z])
        result === nothing || (anchors[objective] = result)
    end
    length(anchors) == length(OBJECTIVES) || return (feasibility=feasibility, solutions=anchors)

    starts = vcat([feasibility.z], [anchors[objective].z for objective in OBJECTIVES])
    solutions = Dict{Symbol, CCUSSolveResult}()
    for objective in OBJECTIVES
        result = best_from_starts(p, objective, starts)
        result === nothing || (solutions[objective] = result)
    end
    return (; feasibility, solutions)
end

function parameter_rows(profile)
    p = profile.p
    return [(
        profile=profile.name,
        evidence_class=profile.evidence,
        rationale=profile.rationale,
        sink_index=k,
        sink=sink_names()[k],
        minimum_CO2_fraction=p.Zmin_k[k],
        capacity=p.Gmax_k[k],
        effective_carbon_loss=p.Mu_k[k],
        safety_index=p.sink_safety[k],
        processing_cost=p.sink_processing_cost[k],
        mean_distance=sum(p.distance[:, k]) / 4,
        mean_compressor_power=sum(p.compressor_power[:, k]) / 4,
        capture_fraction=p.capture_fraction,
        power_carbon_intensity=p.power_carbon_intensity,
        treatment_emission_factor=p.treatment_emission_factor,
        min_algae_flow=p.min_sink1_flow,
    ) for k in 1:6]
end

function solution_row(profile, objective, result, z_names)
    p = profile.p
    structure = system_structure(result.z, p)
    o = result.objective_values
    r = result.residuals
    sink_tuple = NamedTuple{Tuple(Symbol.("sink" .* string.(1:6) .* "_flow"))}(Tuple(structure.sink_flow))
    share_tuple = NamedTuple{Tuple(Symbol.("sink" .* string.(1:6) .* "_share"))}(Tuple(structure.sink_share))
    z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
    return merge((
        scenario=profile.name,
        point_id=profile.name * "__" * String(objective),
        profile=profile.name,
        evidence_class=profile.evidence,
        objective=String(objective),
        termination=string(result.termination),
        primal_status=string(result.primal_status),
        solve_time_s=result.solve_time,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        NetCapture=o.NetCapture,
        main_sink_index=structure.main_sink,
        main_sink=sink_names()[structure.main_sink],
        active_sinks=join(sink_names()[structure.active_sinks], ";"),
        active_sink_count=length(structure.active_sinks),
        pretreated_fraction=structure.pretreated_fraction,
        direct_fraction=structure.direct_fraction,
        max_abs_equality=r.max_abs_equality,
        max_abs_source_carbon_diagnostic=r.max_abs_source_carbon_diagnostic,
        max_inequality_violation=r.max_inequality_violation,
    ), sink_tuple, share_tuple, z_tuple)
end

function add_gradient_rows!(objective_rows, equality_rows, inequality_rows,
                            profile, objective, result, gradient_names)
    p = profile.p
    z = result.z
    point_id = profile.name * "__" * String(objective)
    Jobj = ForwardDiff.jacobian(x -> objective_vector(x, p), z)
    Jeq = ForwardDiff.jacobian(x -> equality_values(x, p), z)
    Jineq = ForwardDiff.jacobian(x -> inequality_values(x, p), z)
    eq_values = equality_values(z, p)
    ineq_values = inequality_values(z, p)
    grad_tuple(v) = NamedTuple{Tuple(gradient_names)}(Tuple(Float64.(v)))

    for i in eachindex(objective_names())
        push!(objective_rows, merge((
            scenario=profile.name,
            point_id,
            solve_type=String(objective),
            objective=objective_names()[i],
        ), grad_tuple(Jobj[i, :])))
    end
    for i in eachindex(equality_names())
        push!(equality_rows, merge((
            scenario=profile.name,
            point_id,
            solve_type=String(objective),
            constraint=equality_names()[i],
            residual=eq_values[i],
        ), grad_tuple(Jeq[i, :])))
    end
    for i in eachindex(inequality_names())
        push!(inequality_rows, merge((
            scenario=profile.name,
            point_id,
            solve_type=String(objective),
            constraint=inequality_names()[i],
            value=ineq_values[i],
            active_1e6=ineq_values[i] >= -FEASIBILITY_TOLERANCE,
        ), grad_tuple(Jineq[i, :])))
    end
end

function profile_summary(profile, solutions)
    complete = length(solutions) == length(OBJECTIVES)
    if !complete
        return (
            profile=profile.name,
            evidence_class=profile.evidence,
            accepted_objectives=length(solutions),
            distinct_main_sinks=missing,
            distinct_active_patterns=missing,
            pretreatment_range=missing,
            minimum_sink_share_distance=missing,
            diversity_score=missing,
        )
    end
    structures = [system_structure(solutions[objective].z, profile.p) for objective in OBJECTIVES]
    main_sinks = getfield.(structures, :main_sink)
    active_patterns = [join(s.active_sinks, ";") for s in structures]
    pretreatment = getfield.(structures, :pretreated_fraction)
    shares = getfield.(structures, :sink_share)
    pair_distances = [sum(abs.(shares[i] .- shares[j])) / 2 for i in 1:2 for j in (i + 1):3]
    distinct_main = length(unique(main_sinks))
    distinct_patterns = length(unique(active_patterns))
    pretreatment_range = maximum(pretreatment) - minimum(pretreatment)
    minimum_distance = minimum(pair_distances)
    score = 10 * distinct_main + 2 * distinct_patterns + 5 * minimum_distance + 5 * pretreatment_range
    return (
        profile=profile.name,
        evidence_class=profile.evidence,
        accepted_objectives=length(solutions),
        distinct_main_sinks=distinct_main,
        distinct_active_patterns=distinct_patterns,
        pretreatment_range,
        minimum_sink_share_distance=minimum_distance,
        diversity_score=score,
    )
end

function main()
    mkpath(RESULT_DIR)
    profiles = parameter_profiles()
    parameter_output = NamedTuple[]
    solution_output = NamedTuple[]
    summary_output = NamedTuple[]
    objective_gradients = NamedTuple[]
    equality_gradients = NamedTuple[]
    inequality_gradients = NamedTuple[]
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))
    gradient_names = Symbol.("dz" .* string.(eachindex(variable_names())))

    for profile in profiles
        append!(parameter_output, parameter_rows(profile))
        solved = solve_profile(profile.p)
        for objective in OBJECTIVES
            haskey(solved.solutions, objective) || continue
            result = solved.solutions[objective]
            push!(solution_output, solution_row(profile, objective, result, z_names))
            add_gradient_rows!(objective_gradients, equality_gradients, inequality_gradients,
                               profile, objective, result, gradient_names)
        end
        push!(summary_output, profile_summary(profile, solved.solutions))
        println(profile.name, ": ", length(solved.solutions), "/3 objectives accepted")
    end

    CSV.write(joinpath(RESULT_DIR, "diversity_parameters.csv"), DataFrame(parameter_output))
    CSV.write(joinpath(RESULT_DIR, "diversity_objective_solutions.csv"), DataFrame(solution_output))
    CSV.write(joinpath(RESULT_DIR, "diversity_candidate_summary.csv"), DataFrame(summary_output))
    CSV.write(joinpath(RESULT_DIR, "diversity_objective_gradients.csv"), DataFrame(objective_gradients))
    CSV.write(joinpath(RESULT_DIR, "diversity_equality_gradients.csv"), DataFrame(equality_gradients))
    CSV.write(joinpath(RESULT_DIR, "diversity_inequality_gradients.csv"), DataFrame(inequality_gradients))
    open(joinpath(RESULT_DIR, "diversity_run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "julia_version = ", VERSION)
        println(io, "profiles = ", length(profiles))
        println(io, "accepted_objective_solutions = ", length(solution_output))
        println(io, "feasibility_tolerance = ", FEASIBILITY_TOLERANCE)
        println(io, "active_sink_flow_tolerance = 1.0")
        println(io, "optimization_note = deterministic cross-start Ipopt; local NLP optima, not global certificates")
    end
end

main()
