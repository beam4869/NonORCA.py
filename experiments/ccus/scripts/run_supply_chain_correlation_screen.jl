using CSV
using DataFrames
using Dates
using ForwardDiff
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const OBJECTIVES = (:TAC, :TotEmiss, :ISI)
const FEASIBILITY_TOLERANCE = 1.0e-6
const CONVERGED = (
    MOI.OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_OPTIMAL,
    MOI.ALMOST_LOCALLY_SOLVED,
)
const PRIMAL_OK = (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)

function accepted(result)
    result.z === nothing && return false
    result.termination in CONVERGED || return false
    result.primal_status in PRIMAL_OK || return false
    r = result.residuals
    return r.max_abs_equality <= FEASIBILITY_TOLERANCE &&
           r.max_abs_source_carbon_diagnostic <= FEASIBILITY_TOLERANCE &&
           r.max_inequality_violation <= FEASIBILITY_TOLERANCE
end

function candidate_profiles()
    c1 = supply_chain_condition1_parameters()
    energy_mild = (
        power_price=0.04,
        power_carbon_intensity=0.55,
        treatment_emission_factor=0.050,
    )
    energy_high = (
        power_price=0.08,
        power_carbon_intensity=0.732,
        treatment_emission_factor=0.0676,
    )
    loss_mild = (
        Mu_k=Float64[0.24, 0.30, 0.01, 0.24, 0.03, 0.32],
    )
    loss_strong = (
        Mu_k=Float64[0.30, 0.45, 0.02, 0.30, 0.00, 0.40],
    )
    safety_decoupled = (
        sink_safety=Float64[10, 2, 40, 38, 32, 36],
        transport_safety=4.0,
        treatment_safety=4.0,
    )
    safety_extreme = (
        sink_safety=Float64[15, 1, 60, 55, 50, 55],
        transport_safety=1.0,
        treatment_safety=1.0,
    )
    capacity_balanced = (
        Gmax_k=Float64[250, 1000, 650, 500, 700, 500],
    )
    capacity_tight = (
        Gmax_k=Float64[200, 850, 550, 450, 600, 450],
    )
    supply_shift = (
        M_s=Float64[357, 1050, 350, 2800],
    )
    matched_treatment_cost = fill(
        energy_high.treatment_emission_factor * 1000 * energy_high.power_price /
        energy_high.power_carbon_intensity,
        4,
    )
    shared_operations_mild = (
        power_price=energy_high.power_price,
        power_carbon_intensity=energy_high.power_carbon_intensity,
        treatment_emission_factor=energy_high.treatment_emission_factor,
        treatment_cost=matched_treatment_cost,
        capital_recovery_factor=0.08,
        Mu_k=Float64[0.08, 0.10, 0.00, 0.08, 0.02, 0.10],
        sink_processing_cost=0.25 .* c1.sink_processing_cost,
        source_capture_fraction=fill(0.995, 4),
    )
    shared_operations_moderate = (
        power_price=energy_high.power_price,
        power_carbon_intensity=energy_high.power_carbon_intensity,
        treatment_emission_factor=energy_high.treatment_emission_factor,
        treatment_cost=matched_treatment_cost,
        capital_recovery_factor=0.03,
        Mu_k=Float64[0.03, 0.04, 0.00, 0.03, 0.01, 0.04],
        sink_processing_cost=zeros(6),
        source_capture_fraction=fill(0.999, 4),
    )
    shared_operations_strong = (
        power_price=energy_high.power_price,
        power_carbon_intensity=energy_high.power_carbon_intensity,
        treatment_emission_factor=energy_high.treatment_emission_factor,
        treatment_cost=matched_treatment_cost,
        capital_recovery_factor=0.005,
        Mu_k=zeros(6),
        sink_processing_cost=zeros(6),
        source_capture_fraction=ones(4),
    )
    operating_alignment_base = (
        power_price=energy_high.power_price,
        power_carbon_intensity=energy_high.power_carbon_intensity,
        treatment_emission_factor=energy_high.treatment_emission_factor,
        treatment_cost=matched_treatment_cost,
        Mu_k=zeros(6),
        sink_processing_cost=zeros(6),
        source_capture_fraction=ones(4),
        pump_power=zeros(size(c1.pump_power)),
    )
    operating_scaled(scale, recovery) = merge(
        operating_alignment_base,
        (
            power_price=energy_high.power_price * scale,
            treatment_cost=matched_treatment_cost .* scale,
            capital_recovery_factor=recovery,
        ),
    )
    carbon_cost_safety = (
        sink_safety=Float64[2, 1, 60, 2, 60, 1],
        transport_safety=1.0,
        treatment_safety=1.0,
    )
    function carbon_cost_alignment(scale; capacities=nothing, supplies=nothing)
        ratio = 1000 * energy_high.power_price / energy_high.power_carbon_intensity
        changes = (
            power_price=energy_high.power_price * scale,
            power_carbon_intensity=energy_high.power_carbon_intensity,
            treatment_emission_factor=energy_high.treatment_emission_factor,
            treatment_cost=matched_treatment_cost .* scale,
            capital_recovery_factor=0.15,
            Mu_k=loss_strong.Mu_k,
            sink_processing_cost=ratio .* scale .* loss_strong.Mu_k,
            source_capture_fraction=ones(4),
            pump_power=zeros(size(c1.pump_power)),
        )
        capacities === nothing || (changes = merge(changes, capacities))
        supplies === nothing || (changes = merge(changes, supplies))
        return merge(changes, carbon_cost_safety)
    end

    make(name, mechanism, changes) = (
        name=name,
        mechanism=mechanism,
        p=with_parameters(c1; changes...),
    )
    combine(parts...) = merge(parts...)

    return [
        (
            name="supply_chain_condition1",
            mechanism="Frozen original direct-use-expansion supply chain.",
            p=c1,
        ),
        make("energy_mild", "Moderately couples energy cost and power-related emissions.", energy_mild),
        make("energy_high", "Strongly couples energy cost and power-related emissions.", energy_high),
        make("loss_mild", "Makes the low-cost Urea route lower-loss while retaining saline storage.", loss_mild),
        make("loss_strong", "Strong economic-carbon alignment across Urea and competing sinks.", loss_strong),
        make("safety_decoupled", "Makes ISI prefer Greenhouse while penalizing the low-loss sinks.", safety_decoupled),
        make("capacity_balanced", "Rebalances sink/treatment capacities without changing objective coefficients.", capacity_balanced),
        make("capacity_tight", "Tightens all preferred sink/treatment capacities.", capacity_tight),
        make("supply_shift", "Reduces availability of the lower-concentration source streams.", supply_shift),
        make("energy_high_loss_mild", "High energy coupling plus mild sink loss alignment.", combine(energy_high, loss_mild)),
        make("energy_high_loss_strong", "High energy coupling plus strong sink loss alignment.", combine(energy_high, loss_strong)),
        make("loss_mild_safety", "Mild loss alignment plus ISI-decoupled safety coefficients.", combine(loss_mild, safety_decoupled)),
        make("loss_strong_safety", "Strong loss alignment plus ISI-decoupled safety coefficients.", combine(loss_strong, safety_decoupled)),
        make("energy_high_safety", "High energy coupling plus ISI-decoupled safety coefficients.", combine(energy_high, safety_decoupled)),
        make("energy_high_loss_mild_safety", "Energy, carbon-loss, and safety mechanisms combined.", combine(energy_high, loss_mild, safety_decoupled)),
        make("energy_high_loss_strong_safety", "Strong energy-carbon alignment with safety decoupling.", combine(energy_high, loss_strong, safety_decoupled)),
        make("combined_balanced_capacity", "Strong alignment with rebalanced sink/treatment capacities.", combine(energy_high, loss_strong, safety_decoupled, capacity_balanced)),
        make("combined_tight_capacity", "Strong alignment with tight sink/treatment capacities.", combine(energy_high, loss_strong, safety_decoupled, capacity_tight)),
        make("combined_supply_shift", "Strong alignment under changed source availability.", combine(energy_high, loss_strong, safety_decoupled, supply_shift)),
        make("combined_balanced_supply", "Strong alignment with capacity and source-availability changes.", combine(energy_high, loss_strong, safety_decoupled, capacity_balanced, supply_shift)),
        make("combined_tight_supply", "Strong alignment under tight capacities and changed source availability.", combine(energy_high, loss_strong, safety_decoupled, capacity_tight, supply_shift)),
        make("combined_capture_50", "Strong alignment, balanced capacities, and a 50% capture target.", combine(energy_high, loss_strong, safety_decoupled, capacity_balanced, (capture_fraction=0.50,))),
        make("combined_capture_60", "Strong alignment, expanded capacities, and a 60% capture target.", combine(
            energy_high,
            loss_strong,
            safety_decoupled,
            (Gmax_k=Float64[300, 1300, 850, 700, 900, 700], capture_fraction=0.60),
        )),
        make("shared_operations_mild", "Makes treatment and operating energy shared cost-emission drivers while retaining moderate capital and sink terms.", shared_operations_mild),
        make("shared_operations_moderate", "Makes treatment and operating energy dominant shared drivers with small residual sink losses.", shared_operations_moderate),
        make("shared_operations_strong", "Mechanism-limit case in which TAC and emissions are almost entirely shared operating terms.", shared_operations_strong),
        make("shared_mild_safety", "Mild shared-operations case with an independently low-ISI Greenhouse route.", combine(shared_operations_mild, safety_decoupled)),
        make("shared_moderate_safety", "Moderate shared-operations case with safety decoupling.", combine(shared_operations_moderate, safety_decoupled)),
        make("shared_strong_safety", "Strong shared-operations case with safety decoupling.", combine(shared_operations_strong, safety_decoupled)),
        make("shared_moderate_safety_extreme", "Moderate shared operations with a stronger independent ISI contrast.", combine(shared_operations_moderate, safety_extreme)),
        make("shared_strong_safety_extreme", "Strong shared operations with a stronger independent ISI contrast.", combine(shared_operations_strong, safety_extreme)),
        make("shared_moderate_capacity", "Moderate shared operations under rebalanced sink/treatment capacities.", combine(shared_operations_moderate, safety_extreme, capacity_balanced)),
        make("shared_moderate_tight_capacity", "Moderate shared operations under tight sink/treatment capacities.", combine(shared_operations_moderate, safety_extreme, capacity_tight)),
        make("shared_moderate_supply", "Moderate shared operations under changed source availability.", combine(shared_operations_moderate, safety_extreme, supply_shift)),
        make("shared_moderate_capacity_supply", "Moderate shared operations with both capacity and source changes.", combine(shared_operations_moderate, safety_extreme, capacity_balanced, supply_shift)),
        make("shared_strong_capacity_supply", "Mechanism-limit shared operations with capacity and source changes.", combine(shared_operations_strong, safety_extreme, capacity_balanced, supply_shift)),
        make("operating_alignment_crf_01", "Operating-alignment mechanism with a 0.01 capital recovery factor.", combine(operating_alignment_base, (capital_recovery_factor=0.01,))),
        make("operating_alignment_crf_001", "Operating-alignment mechanism with a 0.001 capital recovery factor.", combine(operating_alignment_base, (capital_recovery_factor=0.001,))),
        make("operating_alignment_limit", "Mechanism limit with zero annualized capital contribution.", combine(operating_alignment_base, (capital_recovery_factor=0.0,))),
        make("operating_alignment_001_safety", "Near-operating-limit case with an independent low-ISI route.", combine(operating_alignment_base, safety_extreme, (capital_recovery_factor=0.001,))),
        make("operating_alignment_limit_safety", "Operating-limit case with an independent low-ISI route.", combine(operating_alignment_base, safety_extreme, (capital_recovery_factor=0.0,))),
        make("operating_alignment_001_capacity", "Near-operating-limit case with rebalanced sink/treatment capacities.", combine(operating_alignment_base, safety_extreme, capacity_balanced, (capital_recovery_factor=0.001,))),
        make("operating_alignment_limit_capacity", "Operating-limit case with rebalanced sink/treatment capacities.", combine(operating_alignment_base, safety_extreme, capacity_balanced, (capital_recovery_factor=0.0,))),
        make("operating_alignment_001_capacity_supply", "Near-operating-limit case with capacity and source changes.", combine(operating_alignment_base, safety_extreme, capacity_balanced, supply_shift, (capital_recovery_factor=0.001,))),
        make("operating_alignment_limit_capacity_supply", "Operating-limit case with capacity and source changes.", combine(operating_alignment_base, safety_extreme, capacity_balanced, supply_shift, (capital_recovery_factor=0.0,))),
        make("operating_scale_10", "Tenfold shared operating and treatment costs with the original capital recovery factor.", operating_scaled(10.0, 0.15)),
        make("operating_scale_100", "Hundredfold shared operating and treatment costs with the original capital recovery factor.", operating_scaled(100.0, 0.15)),
        make("operating_scale_1000", "Thousandfold shared operating and treatment costs with the original capital recovery factor.", operating_scaled(1000.0, 0.15)),
        make("operating_scale_10000", "Mechanism-limit high operating costs with the original capital recovery factor.", operating_scaled(10000.0, 0.15)),
        make("operating_scale_10_crf01", "Tenfold shared operating costs with a 0.01 capital recovery factor.", operating_scaled(10.0, 0.01)),
        make("operating_scale_100_crf01", "Hundredfold shared operating costs with a 0.01 capital recovery factor.", operating_scaled(100.0, 0.01)),
        make("operating_scale_1000_crf01", "Thousandfold shared operating costs with a 0.01 capital recovery factor.", operating_scaled(1000.0, 0.01)),
        make("operating_scale_100_safety_capacity", "Hundredfold shared operating costs with safety and capacity changes.", combine(operating_scaled(100.0, 0.15), safety_extreme, capacity_balanced)),
        make("operating_scale_1000_safety_capacity", "Thousandfold shared operating costs with safety and capacity changes.", combine(operating_scaled(1000.0, 0.15), safety_extreme, capacity_balanced)),
        make("operating_scale_100_crf01_capacity_supply", "Hundredfold shared operating costs, lower capital recovery, capacity changes, and source changes.", combine(operating_scaled(100.0, 0.01), safety_extreme, capacity_balanced, supply_shift)),
        make("carbon_cost_alignment_1", "Aligns sink processing cost with carbon loss at the base operating-cost scale.", carbon_cost_alignment(1.0)),
        make("carbon_cost_alignment_10", "Carbon-loss-aligned sink cost with tenfold shared operating terms.", carbon_cost_alignment(10.0)),
        make("carbon_cost_alignment_100", "Carbon-loss-aligned sink cost with hundredfold shared operating terms.", carbon_cost_alignment(100.0)),
        make("carbon_cost_alignment_1000", "Carbon-loss-aligned sink cost with thousandfold shared operating terms.", carbon_cost_alignment(1000.0)),
        make("carbon_cost_alignment_1_capacity", "Base-scale carbon-loss-aligned costs with rebalanced capacities.", carbon_cost_alignment(1.0; capacities=capacity_balanced)),
        make("carbon_cost_alignment_10_capacity", "Carbon-loss-aligned costs with rebalanced sink/treatment capacities.", carbon_cost_alignment(10.0; capacities=capacity_balanced)),
        make("carbon_cost_alignment_100_capacity", "Hundredfold carbon-loss-aligned costs with rebalanced capacities.", carbon_cost_alignment(100.0; capacities=capacity_balanced)),
        make("carbon_cost_alignment_10_capacity_supply", "Carbon-loss-aligned costs under capacity and source changes.", carbon_cost_alignment(10.0; capacities=capacity_balanced, supplies=supply_shift)),
        make("carbon_cost_alignment_1_capacity_supply", "Base-scale carbon-loss-aligned costs under capacity and source changes.", carbon_cost_alignment(1.0; capacities=capacity_balanced, supplies=supply_shift)),
        make("carbon_cost_alignment_100_capacity_supply", "Hundredfold carbon-loss-aligned costs under capacity and source changes.", carbon_cost_alignment(100.0; capacities=capacity_balanced, supplies=supply_shift)),
    ]
end

function distinct_starts(starts)
    output = Vector{Vector{Float64}}()
    for start in starts
        start === nothing && continue
        candidate = Float64.(start)
        if all(maximum(abs.(candidate .- existing)) > 1.0e-8 for existing in output)
            push!(output, candidate)
        end
    end
    return output
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
    accepted(feasibility) || return (;
        feasibility,
        solutions=Dict{Symbol, CCUSSolveResult}(),
    )

    preliminary = Dict{Symbol, CCUSSolveResult}()
    for objective in OBJECTIVES
        result = best_from_starts(p, objective, [feasibility.z])
        result === nothing || (preliminary[objective] = result)
    end
    length(preliminary) == length(OBJECTIVES) || return (;
        feasibility,
        solutions=preliminary,
    )

    starts = vcat([feasibility.z], [preliminary[o].z for o in OBJECTIVES])
    solutions = Dict{Symbol, CCUSSolveResult}()
    for objective in OBJECTIVES
        result = best_from_starts(p, objective, starts)
        result === nothing || (solutions[objective] = result)
    end
    return (; feasibility, solutions)
end

function parameter_rows(profile)
    p = profile.p
    rows = NamedTuple[]
    for k in eachindex(sink_names())
        push!(rows, (
            scenario=profile.name,
            mechanism=profile.mechanism,
            sink_index=k,
            sink=sink_names()[k],
            capacity=p.Gmax_k[k],
            minimum_CO2_fraction=p.Zmin_k[k],
            effective_carbon_loss=p.Mu_k[k],
            safety_index=p.sink_safety[k],
            processing_cost=p.sink_processing_cost[k],
            mean_distance=sum(p.distance[:, k]) / size(p.distance, 1),
            mean_compressor_power=sum(p.compressor_power[:, k]) / size(p.compressor_power, 1),
            mean_pump_power=sum(p.pump_power[:, k]) / size(p.pump_power, 1),
            source1_capacity=p.M_s[1],
            source2_capacity=p.M_s[2],
            source3_capacity=p.M_s[3],
            source4_capacity=p.M_s[4],
            capture_fraction=p.capture_fraction,
            power_price=p.power_price,
            power_carbon_intensity=p.power_carbon_intensity,
            treatment_emission_factor=p.treatment_emission_factor,
            treatment_cost_source1=p.treatment_cost[1],
            treatment_cost_source2=p.treatment_cost[2],
            treatment_cost_source3=p.treatment_cost[3],
            treatment_cost_source4=p.treatment_cost[4],
            source_capture_fraction1=p.source_capture_fraction[1],
            source_capture_fraction2=p.source_capture_fraction[2],
            source_capture_fraction3=p.source_capture_fraction[3],
            source_capture_fraction4=p.source_capture_fraction[4],
            transport_safety=p.transport_safety,
            treatment_safety=p.treatment_safety,
        ))
    end
    return rows
end

function solution_row(profile, objective, result, z_names)
    o = result.objective_values
    r = result.residuals
    structure = system_structure(result.z, profile.p)
    z_tuple = NamedTuple{Tuple(z_names)}(Tuple(result.z))
    sink_tuple = NamedTuple{Tuple(Symbol.("sink" .* string.(1:6) .* "_share"))}(
        Tuple(structure.sink_share),
    )
    return merge((
        scenario=profile.name,
        mechanism=profile.mechanism,
        point_id=profile.name * "__" * String(objective),
        solve_type=String(objective),
        termination=string(result.termination),
        primal_status=string(result.primal_status),
        solve_time_s=result.solve_time,
        TAC=o.TAC,
        TotEmiss=o.TotEmiss,
        ISI=o.ISI,
        NetCapture=o.NetCapture,
        main_sink=sink_names()[structure.main_sink],
        active_sinks=join(sink_names()[structure.active_sinks], ";"),
        pretreated_fraction=structure.pretreated_fraction,
        direct_fraction=structure.direct_fraction,
        max_abs_equality=r.max_abs_equality,
        max_abs_source_carbon_diagnostic=r.max_abs_source_carbon_diagnostic,
        max_inequality_violation=r.max_inequality_violation,
    ), sink_tuple, z_tuple)
end

function gradient_tuple(values)
    names = Tuple(Symbol.("dz" .* string.(eachindex(values))))
    return NamedTuple{names}(Tuple(Float64.(values)))
end

function add_gradient_rows!(objective_rows, equality_rows, inequality_rows, profile, result)
    p = profile.p
    z = result.z
    point_id = profile.name * "__" * String(result.objective)
    Jobj = ForwardDiff.jacobian(x -> objective_vector(x, p), z)
    Jeq = ForwardDiff.jacobian(x -> equality_values(x, p), z)
    Jineq = ForwardDiff.jacobian(x -> inequality_values(x, p), z)
    eq_values = equality_values(z, p)
    ineq_values = inequality_values(z, p)

    for i in eachindex(objective_names())
        push!(objective_rows, merge((
            scenario=profile.name,
            point_id,
            solve_type=String(result.objective),
            objective=objective_names()[i],
        ), gradient_tuple(Jobj[i, :])))
    end
    for i in eachindex(equality_names())
        push!(equality_rows, merge((
            scenario=profile.name,
            point_id,
            solve_type=String(result.objective),
            constraint=equality_names()[i],
            residual=eq_values[i],
        ), gradient_tuple(Jeq[i, :])))
    end
    for i in eachindex(inequality_names())
        push!(inequality_rows, merge((
            scenario=profile.name,
            point_id,
            solve_type=String(result.objective),
            constraint=inequality_names()[i],
            value=ineq_values[i],
            active_1e6=ineq_values[i] >= -FEASIBILITY_TOLERANCE,
        ), gradient_tuple(Jineq[i, :])))
    end
end

function main()
    mkpath(RESULT_DIR)
    profiles = candidate_profiles()
    parameter_output = NamedTuple[]
    solution_output = NamedTuple[]
    objective_gradients = NamedTuple[]
    equality_gradients = NamedTuple[]
    inequality_gradients = NamedTuple[]
    status_rows = NamedTuple[]
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))

    for profile in profiles
        append!(parameter_output, parameter_rows(profile))
        solved = solve_profile(profile.p)
        for objective in OBJECTIVES
            haskey(solved.solutions, objective) || continue
            result = solved.solutions[objective]
            push!(solution_output, solution_row(profile, objective, result, z_names))
            add_gradient_rows!(
                objective_gradients,
                equality_gradients,
                inequality_gradients,
                profile,
                result,
            )
        end
        push!(status_rows, (
            scenario=profile.name,
            mechanism=profile.mechanism,
            feasible=accepted(solved.feasibility),
            accepted_objectives=length(solved.solutions),
            feasibility_termination=string(solved.feasibility.termination),
            feasibility_max_inequality=isnothing(solved.feasibility.residuals) ?
                missing : solved.feasibility.residuals.max_inequality_violation,
        ))
        println(profile.name, ": ", length(solved.solutions), "/3 objectives accepted")
    end

    CSV.write(
        joinpath(RESULT_DIR, "supply_chain_correlation_screen_parameters.csv"),
        DataFrame(parameter_output),
    )
    CSV.write(
        joinpath(RESULT_DIR, "supply_chain_correlation_screen_solutions.csv"),
        DataFrame(solution_output),
    )
    CSV.write(
        joinpath(RESULT_DIR, "supply_chain_correlation_screen_objective_gradients.csv"),
        DataFrame(objective_gradients),
    )
    CSV.write(
        joinpath(RESULT_DIR, "supply_chain_correlation_screen_equality_gradients.csv"),
        DataFrame(equality_gradients),
    )
    CSV.write(
        joinpath(RESULT_DIR, "supply_chain_correlation_screen_inequality_gradients.csv"),
        DataFrame(inequality_gradients),
    )
    CSV.write(
        joinpath(RESULT_DIR, "supply_chain_correlation_screen_status.csv"),
        DataFrame(status_rows),
    )
    open(joinpath(RESULT_DIR, "supply_chain_correlation_screen_run.txt"), "w") do io
        println(io, "completed_at = ", Dates.now())
        println(io, "condition1 = exact alias of prior direct_use_expansion_parameters")
        println(io, "candidate_scenarios = ", length(profiles))
        println(io, "accepted_objective_solutions = ", length(solution_output))
        println(io, "screening_method = three single-objective optima with deterministic cross-start Ipopt")
        println(io, "global_optimality = not claimed")
    end
end

main()
