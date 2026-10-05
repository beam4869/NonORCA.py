module CCUSModel

using ForwardDiff
using Ipopt
using JuMP
import MathOptInterface as MOI

export CCUSParameters,
       CCUSSolveResult,
       baseline_parameters,
       build_model,
       co2_generation,
       direct_use_expansion_parameters,
       equality_names,
       equality_values,
       initial_feasible_guess,
       inequality_names,
       inequality_values,
       objective_names,
       objective_values,
       objective_vector,
       residual_summary,
       sink_names,
       solve_ccus,
       source_carbon_residuals,
       supply_chain_condition1_parameters,
       supply_chain_condition2_parameters,
       system_structure,
       variable_names,
       with_parameters

"""Parameters copied from the legacy CCUS scripts, with explicit scenario scalars."""
Base.@kwdef struct CCUSParameters
    M_s::Vector{Float64}
    y_s::Vector{Float64}
    y_s_t::Vector{Float64}
    Zmin_k::Vector{Float64}
    Gmax_k::Vector{Float64}
    Mu_k::Vector{Float64}
    compressor_power::Matrix{Float64}
    pump_power::Matrix{Float64}
    pressure_drop::Matrix{Float64}
    distance::Matrix{Float64}
    capture_efficiency::Float64
    power_carbon_intensity::Float64
    power_price::Float64
    capital_recovery_factor::Float64
    treatment_emission_factor::Float64
    capture_fraction::Float64
    treatment_cost::Vector{Float64}
    transport_safety::Float64
    treatment_safety::Float64
    sink_safety::Vector{Float64}
    temperature::Vector{Float64}
    source_pressure::Vector{Float64}
    sink_pressure::Vector{Float64}
    velocity::Matrix{Float64}
    molecular_mass::Vector{Float64}
    sink_processing_cost::Vector{Float64}
    source_capture_fraction::Vector{Float64}
    min_sink1_flow::Float64
    nonlinear_floor::Float64 = 1.0e-6
end

"""Frozen original supply-chain profile used in the prior Pareto/ORCA analyses."""
function supply_chain_condition1_parameters()
    base = baseline_parameters()
    scale_columns(matrix, factors) = matrix .* reshape(Float64.(factors), 1, :)
    return with_parameters(
        base;
        min_sink1_flow=0.0,
        Zmin_k=Float64[0.06, 0.20, 0.95, 0.999, 0.999, 0.999],
        Gmax_k=Float64[300, 1400, 1400, 756, 900, 913],
        Mu_k=Float64[0.22, 0.16, 0.00, 0.18, 0.12, 0.30],
        sink_safety=Float64[7, 3, 18, 32, 25, 30],
        compressor_power=scale_columns(base.compressor_power, [1.0, 1.0, 1.0, 1.0, 0.18, 1.0]),
        pump_power=scale_columns(base.pump_power, [1.0, 1.0, 1.0, 1.0, 0.18, 1.0]),
        distance=scale_columns(base.distance, [1.0, 0.35, 3.0, 1.5, 0.30, 1.5]),
    )
end

"""Backward-compatible name for the frozen supply-chain condition 1."""
direct_use_expansion_parameters() = supply_chain_condition1_parameters()

"""
Exploratory supply-chain condition 2.

This condition preserves the three objective definitions and couples economic and
carbon performance through carbon-loss-dependent sink processing costs. It also
rebalances sink capacities and changes source availability. Values are scenario
parameters for mechanism testing, not calibrated plant data.
"""
function supply_chain_condition2_parameters()
    c1 = supply_chain_condition1_parameters()
    power_price = 0.08
    power_carbon_intensity = 0.732
    treatment_emission_factor = 0.0676
    cost_to_emission_ratio = 1000 * power_price / power_carbon_intensity
    sink_loss = Float64[0.30, 0.45, 0.02, 0.30, 0.00, 0.40]
    return with_parameters(
        c1;
        M_s=Float64[357, 1050, 350, 2800],
        Gmax_k=Float64[250, 1000, 650, 500, 700, 500],
        Mu_k=sink_loss,
        power_price,
        power_carbon_intensity,
        treatment_emission_factor,
        treatment_cost=fill(treatment_emission_factor * cost_to_emission_ratio, 4),
        sink_processing_cost=cost_to_emission_ratio .* sink_loss,
        source_capture_fraction=ones(4),
        pump_power=zeros(size(c1.pump_power)),
        sink_safety=Float64[2, 1, 60, 2, 60, 1],
        transport_safety=1.0,
        treatment_safety=1.0,
    )
end

"""Return a copy of `p` with only the supplied fields replaced."""
function with_parameters(p::CCUSParameters; kwargs...)
    names = fieldnames(CCUSParameters)
    values = NamedTuple{names}(Tuple(getfield(p, name) for name in names))
    return CCUSParameters(; merge(values, (; kwargs...))...)
end

function baseline_parameters(;
    capture_fraction::Real = 0.40,
    capture_efficiency::Real = 0.90,
    power_carbon_intensity::Real = 0.366,
    power_price::Real = 0.02,
    sink_capacity_scale::Real = 1.0,
    sink_loss_scale::Real = 1.0,
    safety_scale::Real = 1.0,
    min_sink1_flow::Real = 100.0,
)
    capacity_scale = Float64(sink_capacity_scale)
    loss_scale = Float64(sink_loss_scale)
    return CCUSParameters(
        M_s = Float64[357, 1260, 399, 3426],
        y_s = Float64[1.0, 0.44, 0.27, 0.07],
        y_s_t = Float64[1.0, 0.999, 0.999, 0.999],
        Zmin_k = Float64[0.06, 0.94, 0.94, 0.999, 0.999, 0.999],
        Gmax_k = capacity_scale .* Float64[126, 508, 3168, 756, 543, 913],
        Mu_k = clamp.(loss_scale .* Float64[0.28, 0.50, 0.0, 0.17, 0.11, 0.34], 0.0, 0.999),
        compressor_power = Float64[
            0.43 2.22 4.37 4.37 4.37 4.37;
            0.49 2.23 4.37 4.37 4.37 4.37;
            0.62 2.28 4.37 4.37 4.37 4.37;
            0.58 2.28 4.37 4.37 4.37 4.37
        ],
        pump_power = Float64[
            0.0 0.0 0.204 0.017 0.176 0.204;
            0.0 0.0 0.204 0.017 0.176 0.204;
            0.0 0.0 0.205 0.018 0.176 0.203;
            0.0 0.0 0.205 0.018 0.175 0.203
        ],
        pressure_drop = Float64[
            56.39 832.13 49.51 49.51 50.82 51.15;
            67.87 843.61 60.98 60.98 62.30 62.62;
            90.82 896.07 96.72 96.72 29.84 16.72;
            82.95 888.20 88.85 88.85 21.64 26.89
        ],
        distance = Float64[
            1.72 25.38 1.51 1.51 1.55 1.56;
            2.07 25.73 1.86 1.86 1.90 1.91;
            2.77 27.33 2.95 2.95 0.91 0.51;
            2.53 27.09 2.71 2.71 0.66 0.82
        ],
        capture_efficiency = Float64(capture_efficiency),
        power_carbon_intensity = Float64(power_carbon_intensity),
        power_price = Float64(power_price),
        capital_recovery_factor = 0.15,
        treatment_emission_factor = 0.0338,
        capture_fraction = Float64(capture_fraction),
        treatment_cost = Float64[0, 29, 43, 35],
        transport_safety = 12.0 * Float64(safety_scale),
        treatment_safety = 19.0 * Float64(safety_scale),
        sink_safety = Float64(safety_scale) .* Float64[7, 7, 8, 36, 25, 34],
        temperature = fill(298.0, 4),
        source_pressure = fill(101.0, 4),
        sink_pressure = Float64[101, 101, 8080, 14140, 15198, 15198],
        velocity = fill(20.0, 4, 6),
        molecular_mass = fill(44.0, 4),
        sink_processing_cost = Float64[-7, -5, 9, -20, -17, -26],
        source_capture_fraction = fill(0.99, 4),
        min_sink1_flow = Float64(min_sink1_flow),
    )
end

struct VariableLayout
    R::UnitRange{Int}
    T::UnitRange{Int}
    U::UnitRange{Int}
    V::UnitRange{Int}
    F::UnitRange{Int}
    n::Int
end

function variable_layout()
    rR = 1:4
    rT = (last(rR) + 1):(last(rR) + 24)
    rU = (last(rT) + 1):(last(rT) + 24)
    rV = (last(rU) + 1):(last(rU) + 24)
    rF = (last(rV) + 1):(last(rV) + 6)
    return VariableLayout(rR, rT, rU, rV, rF, last(rF))
end

function variable_names()
    names = String[]
    append!(names, ["R[$s]" for s in 1:4])
    append!(names, ["T[$s,$k]" for k in 1:6 for s in 1:4])
    append!(names, ["U[$s,$k]" for k in 1:6 for s in 1:4])
    append!(names, ["V[$s,$k]" for k in 1:6 for s in 1:4])
    append!(names, ["F[$k]" for k in 1:6])
    return names
end

objective_names() = ["TAC", "TotEmiss", "ISI"]
sink_names() = ["Algae", "Greenhouse", "Saline Storage", "Methanol", "Urea", "Acetic Acid"]
co2_generation(p::CCUSParameters) = sum(p.M_s .* p.y_s)

"""Map normalized variables z in [0,1] to physical flows."""
function physical_variables(z::AbstractVector, p::CCUSParameters)
    layout = variable_layout()
    R = p.M_s .* z[layout.R]
    Tflow = reshape(z[layout.T], 4, 6) .* reshape(p.M_s, 4, 1)
    Uflow = reshape(z[layout.U], 4, 6) .* reshape(p.M_s, 4, 1)
    Vflow = reshape(z[layout.V], 4, 6) .* reshape(p.M_s, 4, 1)
    Yv = [
        (1 - p.capture_efficiency) * p.y_s[s] * p.y_s_t[s] /
        (p.y_s_t[s] - p.capture_efficiency * p.y_s[s])
        for s in 1:4
    ]
    F = p.Gmax_k .* z[layout.F]
    return (; R, Tflow, Uflow, Vflow, Yv, F)
end

"""Summarize the sink allocation and preprocessing structure of a feasible point."""
function system_structure(
    z::AbstractVector,
    p::CCUSParameters;
    active_sink_tolerance::Real = 1.0,
)
    q = physical_variables(z, p)
    total_feed = sum(q.R)
    total_sink_flow = sum(q.F)
    pretreated_feed = sum(q.Tflow) + sum(q.Vflow)
    direct_feed = sum(q.Uflow)
    sink_share = total_sink_flow > 0 ? q.F ./ total_sink_flow : zeros(6)
    return (
        sink_flow = q.F,
        sink_share,
        active_sinks = findall(q.F .> active_sink_tolerance),
        main_sink = argmax(q.F),
        pretreated_fraction = total_feed > 0 ? pretreated_feed / total_feed : 0.0,
        direct_fraction = total_feed > 0 ? direct_feed / total_feed : 0.0,
    )
end

function common_quantities(z::AbstractVector, p::CCUSParameters)
    v = physical_variables(z, p)
    compression_power = 1000 * sum(
        p.compressor_power[s, k] * (v.Tflow[s, k] + v.Uflow[s, k])
        for s in 1:4, k in 1:6
    )
    pump_power = 1000 * 0.8 * sum(
        p.pump_power[s, k] * (v.Tflow[s, k] + v.Uflow[s, k])
        for s in 1:4, k in 1:6
    )
    transport_power = compression_power + pump_power
    fco2 = [
        sum(v.Tflow[s, k] * p.y_s_t[s] + v.Uflow[s, k] * p.y_s[s] for s in 1:4)
        for k in 1:6
    ]
    treatment_emissions = sum(
        v.Tflow[s, k] * p.y_s_t[s] * p.treatment_emission_factor
        for s in 1:4, k in 1:6
    )
    # compressor_power and pump_power are kW-day/year; this converts to kt CO2/year.
    transport_emissions = transport_power * 24 * p.power_carbon_intensity / 1.0e6
    net_capture = sum(fco2[k] * (1 - p.Mu_k[k]) for k in 1:6) -
                  treatment_emissions - transport_emissions
    return (; v..., compression_power, pump_power, transport_power, fco2,
            treatment_emissions, transport_emissions, net_capture)
end

function objective_values(z::AbstractVector, p::CCUSParameters)
    q = common_quantities(z, p)
    eps = p.nonlinear_floor

    treat_cost = sum(
        q.Tflow[s, k] * p.y_s_t[s] * p.treatment_cost[s]
        for s in 1:4, k in 1:6
    )
    cap_compressor = sum(
        158902 * ((p.compressor_power[s, k] * (q.Tflow[s, k] + q.Uflow[s, k]) / 224)^2 + eps^2)^0.42 *
        p.capital_recovery_factor
        for s in 1:4, k in 1:6
    )
    op_compressor = sum(
        24 * p.compressor_power[s, k] * (q.Tflow[s, k] + q.Uflow[s, k]) * p.power_price
        for s in 1:4, k in 1:6
    )
    cap_pump = sum(
        (1.11e3 * p.pump_power[s, k] * (q.Tflow[s, k] + q.Uflow[s, k]) + 70000) *
        p.capital_recovery_factor
        for s in 1:4, k in 1:6
    )
    op_pump = sum(
        24 * 0.8 * p.pump_power[s, k] * (q.Tflow[s, k] + q.Uflow[s, k] + 70000) *
        p.capital_recovery_factor
        for s in 1:4, k in 1:6
    )
    dconst = [
        sqrt(p.velocity[s, k] * p.molecular_mass[s] *
             ((p.sink_pressure[k] - p.source_pressure[s]) + p.pressure_drop[s, k]))
        for s in 1:4, k in 1:6
    ]
    flow_to_sink = [sum(q.Tflow[s, k] + q.Uflow[s, k] for s in 1:4) for k in 1:6]
    diameter = [
        (((4 / pi) * 8.314 * p.temperature[s] * flow_to_sink[k] / dconst[s, k])^2 + eps^2)^0.25
        for s in 1:4, k in 1:6
    ]
    pipe_cost = sum(
        p.distance[s, k] * (95230 * diameter[s, k] + 96904) * p.capital_recovery_factor
        for s in 1:4, k in 1:6
    )
    sink_cost = sum(q.F[k] * p.sink_processing_cost[k] for k in 1:6)
    tac = treat_cost + cap_compressor + op_compressor + cap_pump + op_pump + pipe_cost + sink_cost

    # Corrected carbon accounting: every term is in kt CO2/year.
    source_emissions = sum((1 - p.source_capture_fraction[s]) * q.R[s] * p.y_s[s] for s in 1:4)
    sink_losses = sum(q.fco2[k] * p.Mu_k[k] for k in 1:6)
    total_emissions = source_emissions + q.treatment_emissions + q.transport_emissions + sink_losses

    isi_sink = sum(
        p.sink_safety[k] * sum(q.Tflow[s, k] + q.Uflow[s, k] for s in 1:4) / p.Gmax_k[k]
        for k in 1:6
    )
    isi_transport = p.transport_safety * sum(q.R) / sum(p.M_s)
    isi_treatment = p.treatment_safety * sum(q.Tflow .+ q.Vflow) / sum(p.M_s)
    isi = isi_sink + isi_transport + isi_treatment

    return (; TAC = tac, TotEmiss = total_emissions, ISI = isi, NetCapture = q.net_capture)
end

objective_vector(z::AbstractVector, p::CCUSParameters) = begin
    o = objective_values(z, p)
    [o.TAC, o.TotEmiss, o.ISI]
end

function equality_names()
    names = String[]
    append!(names, ["source_mass[$s]" for s in 1:4])
    append!(names, ["treatment_carbon[$s,$k]" for s in 1:4 for k in 1:6])
    append!(names, ["sink_mass[$k]" for k in 1:6])
    return names
end

"""Dimensionless equality residuals; feasible values are zero."""
function equality_values(z::AbstractVector, p::CCUSParameters)
    q = common_quantities(z, p)
    values = eltype(z)[]
    for s in 1:4
        push!(values, (q.R[s] - sum(q.Tflow[s, :] .+ q.Uflow[s, :] .+ q.Vflow[s, :])) / p.M_s[s])
    end
    for s in 1:4, k in 1:6
        denom = max(p.M_s[s] * p.y_s[s], 1.0)
        push!(values,
              (p.capture_efficiency * (q.Tflow[s, k] + q.Vflow[s, k]) * p.y_s[s] -
               q.Tflow[s, k] * p.y_s_t[s]) / denom)
    end
    for k in 1:6
        push!(values, (q.F[k] - sum(q.Tflow[s, k] + q.Uflow[s, k] for s in 1:4)) / p.Gmax_k[k])
    end
    return values
end

"""Redundant source-carbon equations retained as an independent diagnostic."""
function source_carbon_residuals(z::AbstractVector, p::CCUSParameters)
    q = common_quantities(z, p)
    return [
        begin
            denom = max(p.M_s[s] * p.y_s[s], 1.0)
            rhs = sum(q.Tflow[s, k] * p.y_s_t[s] + q.Uflow[s, k] * p.y_s[s] +
                      q.Vflow[s, k] * q.Yv[s] for k in 1:6)
            (q.R[s] * p.y_s[s] - rhs) / denom
        end
        for s in 1:4
    ]
end

function inequality_names()
    names = ["sink_purity[$k]" for k in 1:6]
    push!(names, "sink1_min_flow")
    push!(names, "capture_target")
    vars = variable_names()
    append!(names, ["lower:$name" for name in vars])
    append!(names, ["upper:$name" for name in vars])
    return names
end

"""Dimensionless inequalities standardized as g(z) <= 0."""
function inequality_values(z::AbstractVector, p::CCUSParameters)
    q = common_quantities(z, p)
    values = eltype(z)[]
    for k in 1:6
        co2_in = sum(q.Tflow[s, k] * p.y_s_t[s] + q.Uflow[s, k] * p.y_s[s] for s in 1:4)
        push!(values, (q.F[k] * p.Zmin_k[k] - co2_in) / p.Gmax_k[k])
    end
    push!(values, (p.min_sink1_flow - q.F[1]) / p.Gmax_k[1])
    push!(values, (co2_generation(p) * p.capture_fraction - q.net_capture) / co2_generation(p))
    append!(values, -z)
    append!(values, z .- 1)
    return values
end

"""Construct an analytically balanced starting point for the baseline topology."""
function initial_feasible_guess(p::CCUSParameters)
    layout = variable_layout()
    z = zeros(layout.n)
    R = zeros(4)
    Tflow = zeros(4, 6)
    Uflow = zeros(4, 6)
    Vflow = zeros(4, 6)

    # Sink 1 minimum is supplied by the high-purity first source.
    Uflow[1, 1] = min(p.min_sink1_flow, p.M_s[1])
    remaining1 = max(p.M_s[1] - Uflow[1, 1], 0.0)
    treated_fraction1 = p.capture_efficiency * p.y_s[1] / p.y_s_t[1]
    Tflow[1, 3] = remaining1 * treated_fraction1
    Vflow[1, 3] = remaining1 - Tflow[1, 3]
    R[1] = Uflow[1, 1] + Tflow[1, 3] + Vflow[1, 3]

    # A partially used second source supplies the remaining capture requirement.
    raw2 = 0.80 * p.M_s[2]
    treated_fraction2 = p.capture_efficiency * p.y_s[2] / p.y_s_t[2]
    Tflow[2, 3] = raw2 * treated_fraction2
    Vflow[2, 3] = raw2 - Tflow[2, 3]
    R[2] = raw2

    F = [sum(Tflow[:, k] .+ Uflow[:, k]) for k in 1:6]

    z[layout.R] .= R ./ p.M_s
    z[layout.T] .= vec(Tflow ./ reshape(p.M_s, 4, 1))
    z[layout.U] .= vec(Uflow ./ reshape(p.M_s, 4, 1))
    z[layout.V] .= vec(Vflow ./ reshape(p.M_s, 4, 1))
    z[layout.F] .= F ./ p.Gmax_k
    return clamp.(z, 0.0, 1.0)
end

function build_model(
    p::CCUSParameters;
    objective::Symbol = :feasibility,
    start::Union{Nothing, AbstractVector} = nothing,
    time_limit::Real = 120.0,
)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_cpu_time", Float64(time_limit))
    set_optimizer_attribute(model, "tol", 1.0e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1.0e-6)
    set_optimizer_attribute(model, "max_iter", 4000)

    layout = variable_layout()
    @variable(model, 0 <= z[1:layout.n] <= 1)
    z0 = isnothing(start) ? initial_feasible_guess(p) : Float64.(start)
    set_start_value.(z, z0)

    @expression(model, R[s=1:4], p.M_s[s] * z[layout.R[s]])
    @expression(model, Tflow[s=1:4, k=1:6], p.M_s[s] * z[layout.T[(k - 1) * 4 + s]])
    @expression(model, Uflow[s=1:4, k=1:6], p.M_s[s] * z[layout.U[(k - 1) * 4 + s]])
    @expression(model, Vflow[s=1:4, k=1:6], p.M_s[s] * z[layout.V[(k - 1) * 4 + s]])
    @expression(model, F[k=1:6], p.Gmax_k[k] * z[layout.F[k]])

    @constraint(model, source_mass[s=1:4],
        (R[s] - sum(Tflow[s, k] + Uflow[s, k] + Vflow[s, k] for k in 1:6)) / p.M_s[s] == 0)
    @constraint(model, treatment_carbon[s=1:4, k=1:6],
        (p.capture_efficiency * (Tflow[s, k] + Vflow[s, k]) * p.y_s[s] -
         Tflow[s, k] * p.y_s_t[s]) / max(p.M_s[s] * p.y_s[s], 1.0) == 0)
    @constraint(model, sink_mass[k=1:6],
        (F[k] - sum(Tflow[s, k] + Uflow[s, k] for s in 1:4)) / p.Gmax_k[k] == 0)
    @constraint(model, sink_purity[k=1:6],
        (F[k] * p.Zmin_k[k] -
         sum(Tflow[s, k] * p.y_s_t[s] + Uflow[s, k] * p.y_s[s] for s in 1:4)) / p.Gmax_k[k] <= 0)
    @constraint(model, sink1_min_flow, (p.min_sink1_flow - F[1]) / p.Gmax_k[1] <= 0)

    @expression(model, compression_power,
        1000 * sum(p.compressor_power[s, k] * (Tflow[s, k] + Uflow[s, k]) for s in 1:4, k in 1:6))
    @expression(model, pump_power,
        1000 * 0.8 * sum(p.pump_power[s, k] * (Tflow[s, k] + Uflow[s, k]) for s in 1:4, k in 1:6))
    @expression(model, transport_power, compression_power + pump_power)
    @expression(model, fco2[k=1:6],
        sum(Tflow[s, k] * p.y_s_t[s] + Uflow[s, k] * p.y_s[s] for s in 1:4))
    @expression(model, treatment_emissions,
        sum(Tflow[s, k] * p.y_s_t[s] * p.treatment_emission_factor for s in 1:4, k in 1:6))
    @expression(model, transport_emissions,
        transport_power * 24 * p.power_carbon_intensity / 1.0e6)
    @expression(model, net_capture,
        sum(fco2[k] * (1 - p.Mu_k[k]) for k in 1:6) - treatment_emissions - transport_emissions)
    @constraint(model, capture_target,
        (co2_generation(p) * p.capture_fraction - net_capture) / co2_generation(p) <= 0)

    eps = p.nonlinear_floor
    @expression(model, treat_cost,
        sum(Tflow[s, k] * p.y_s_t[s] * p.treatment_cost[s] for s in 1:4, k in 1:6))
    @NLexpression(model, cap_compressor[s=1:4, k=1:6],
        158902 * ((p.compressor_power[s, k] * (Tflow[s, k] + Uflow[s, k]) / 224)^2 + eps^2)^0.42 *
        p.capital_recovery_factor)
    @expression(model, op_compressor[s=1:4, k=1:6],
        24 * p.compressor_power[s, k] * (Tflow[s, k] + Uflow[s, k]) * p.power_price)
    @expression(model, cap_pump[s=1:4, k=1:6],
        (1.11e3 * p.pump_power[s, k] * (Tflow[s, k] + Uflow[s, k]) + 70000) *
        p.capital_recovery_factor)
    @expression(model, op_pump[s=1:4, k=1:6],
        24 * 0.8 * p.pump_power[s, k] * (Tflow[s, k] + Uflow[s, k] + 70000) *
        p.capital_recovery_factor)
    dconst = [
        sqrt(p.velocity[s, k] * p.molecular_mass[s] *
             ((p.sink_pressure[k] - p.source_pressure[s]) + p.pressure_drop[s, k]))
        for s in 1:4, k in 1:6
    ]
    @expression(model, flow_to_sink[k=1:6], sum(Tflow[s, k] + Uflow[s, k] for s in 1:4))
    @NLexpression(model, diameter[s=1:4, k=1:6],
        (((4 / pi) * 8.314 * p.temperature[s] * flow_to_sink[k] / dconst[s, k])^2 + eps^2)^0.25)
    @NLexpression(model, pipe_cost,
        sum(p.distance[s, k] * (95230 * diameter[s, k] + 96904) * p.capital_recovery_factor for s in 1:4, k in 1:6))
    @expression(model, sink_cost, sum(F[k] * p.sink_processing_cost[k] for k in 1:6))
    @NLexpression(model, tac,
        treat_cost +
        sum(cap_compressor[s, k] for s in 1:4, k in 1:6) +
        sum(op_compressor[s, k] for s in 1:4, k in 1:6) +
        sum(cap_pump[s, k] for s in 1:4, k in 1:6) +
        sum(op_pump[s, k] for s in 1:4, k in 1:6) +
        pipe_cost + sink_cost)

    @expression(model, source_emissions,
        sum((1 - p.source_capture_fraction[s]) * R[s] * p.y_s[s] for s in 1:4))
    @expression(model, sink_losses, sum(fco2[k] * p.Mu_k[k] for k in 1:6))
    @expression(model, total_emissions,
        source_emissions + treatment_emissions + transport_emissions + sink_losses)
    @expression(model, isi,
        sum(p.sink_safety[k] * sum(Tflow[s, k] + Uflow[s, k] for s in 1:4) / p.Gmax_k[k] for k in 1:6) +
        p.transport_safety * sum(R) / sum(p.M_s) +
        p.treatment_safety * sum(Tflow[s, k] + Vflow[s, k] for s in 1:4, k in 1:6) / sum(p.M_s))

    if objective == :feasibility
        @objective(model, Min, sum((z[i] - z0[i])^2 for i in 1:layout.n))
    elseif objective == :TAC
        @NLobjective(model, Min, tac / 1.0e7)
    elseif objective == :TotEmiss
        @objective(model, Min, total_emissions / co2_generation(p))
    elseif objective == :ISI
        @objective(model, Min, isi / 50.0)
    elseif objective == :MaxCapture
        @objective(model, Max, net_capture / co2_generation(p))
    else
        error("Unknown objective: $objective")
    end

    return (; model, z, tac, total_emissions, isi, net_capture)
end

Base.@kwdef struct CCUSSolveResult
    objective::Symbol
    termination::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    z::Union{Nothing, Vector{Float64}}
    solve_time::Float64
    objective_values::Union{Nothing, NamedTuple}
    residuals::Union{Nothing, NamedTuple}
end

function residual_summary(z::AbstractVector, p::CCUSParameters)
    eq = equality_values(z, p)
    ineq = inequality_values(z, p)
    return (
        max_abs_equality = maximum(abs, eq),
        max_abs_source_carbon_diagnostic = maximum(abs, source_carbon_residuals(z, p)),
        max_inequality_violation = max(maximum(ineq), 0.0),
        active_inequalities_1e6 = count(v -> v >= -1.0e-6, ineq),
        worst_equality_index = argmax(abs.(eq)),
        worst_inequality_index = argmax(ineq),
    )
end

function solve_ccus(
    p::CCUSParameters;
    objective::Symbol = :feasibility,
    start::Union{Nothing, AbstractVector} = nothing,
    time_limit::Real = 120.0,
)
    bm = build_model(p; objective, start, time_limit)
    elapsed = @elapsed optimize!(bm.model)
    term = termination_status(bm.model)
    pstat = primal_status(bm.model)
    if !has_values(bm.model)
        return CCUSSolveResult(; objective, termination=term, primal_status=pstat,
                               z=nothing, solve_time=elapsed,
                               objective_values=nothing, residuals=nothing)
    end
    zval = value.(bm.z)
    return CCUSSolveResult(; objective, termination=term, primal_status=pstat,
                           z=zval, solve_time=elapsed,
                           objective_values=objective_values(zval, p),
                           residuals=residual_summary(zval, p))
end

end
