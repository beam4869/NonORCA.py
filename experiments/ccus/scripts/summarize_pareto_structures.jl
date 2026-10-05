using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

const RESULT_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const PROFILE = get(ENV, "CCUS_PARETO_PROFILE", "baseline")
const PREFIX = PROFILE == "baseline" ? "" : PROFILE * "_"

function profile_parameters()
    PROFILE == "baseline" && return baseline_parameters()
    PROFILE == "direct_use_expansion" && return direct_use_expansion_parameters()
    PROFILE == "supply_chain_condition1" && return supply_chain_condition1_parameters()
    PROFILE == "supply_chain_condition2" && return supply_chain_condition2_parameters()
    error("Unknown CCUS_PARETO_PROFILE: $PROFILE")
end

function main()
    p = profile_parameters()
    frontier_path = joinpath(RESULT_DIR, PREFIX * "pareto_frontier.csv")
    frontier = CSV.read(frontier_path, DataFrame)
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))
    rows = NamedTuple[]
    for row in eachrow(frontier)
        z = Float64[row[name] for name in z_names]
        structure = system_structure(z, p)
        sink_flows = NamedTuple{
            Tuple(Symbol.("sink" .* string.(1:6) .* "_flow"))
        }(Tuple(structure.sink_flow))
        sink_shares = NamedTuple{
            Tuple(Symbol.("sink" .* string.(1:6) .* "_share"))
        }(Tuple(structure.sink_share))
        push!(rows, merge((
            frontier_id=row.frontier_id,
            TAC=row.TAC,
            TotEmiss=row.TotEmiss,
            ISI=row.ISI,
            TAC_normalized=row.TAC_normalized,
            TotEmiss_normalized=row.TotEmiss_normalized,
            ISI_normalized=row.ISI_normalized,
            main_sink=sink_names()[structure.main_sink],
            active_sinks=join(sink_names()[structure.active_sinks], ";"),
            active_sink_count=length(structure.active_sinks),
            pretreated_fraction=clamp(structure.pretreated_fraction, 0.0, 1.0),
            direct_fraction=clamp(structure.direct_fraction, 0.0, 1.0),
        ), sink_flows, sink_shares))
    end
    output = DataFrame(rows)
    CSV.write(
        joinpath(RESULT_DIR, PREFIX * "pareto_system_structures.csv"), output,
    )
    println(combine(groupby(output, :main_sink), nrow => :frontier_points))
    println(describe(output[:, [:pretreated_fraction, :direct_fraction]]))
end

main()
