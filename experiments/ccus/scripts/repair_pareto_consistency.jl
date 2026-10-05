using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "run_pareto_epsilon.jl"))

function row_start(row, z_names)
    return Float64[row[name] for name in z_names]
end

function accepted_witness(raw, method, index_a, index_b=nothing)
    subset = raw[raw.method .== method, :]
    if isnothing(index_b)
        subset = subset[subset.index_a .>= index_a, :]
        isempty(subset) && return nothing
        distances = subset.index_a .- index_a
    else
        subset = subset[(subset.index_a .>= index_a) .&
                        (subset.index_b .>= index_b), :]
        isempty(subset) && return nothing
        distances = (subset.index_a .- index_a) .+
                    (subset.index_b .- index_b)
    end
    return subset[argmin(distances), :]
end

function consistency_holes(log, raw, two_dimensional)
    holes = Int[]
    for row_index in axes(log, 1)
        row = log[row_index, :]
        row.accepted && continue
        index_b = two_dimensional ? Int(row.index_b) : nothing
        witness = accepted_witness(
            raw, String(row.method), Int(row.index_a), index_b,
        )
        isnothing(witness) || push!(holes, row_index)
    end
    return holes
end

function repair_rows!(raw, log, p, lower, ranges, anchors, z_names;
                      two_dimensional)
    repaired = 0
    passes = 0
    while true
        holes = consistency_holes(log, raw, two_dimensional)
        isempty(holes) && break
        passes += 1
        progress = 0
        for (hole_number, row_index) in enumerate(holes)
            row = log[row_index, :]
            method = Symbol(row.method)
            index_b = two_dimensional ? Int(row.index_b) : nothing
            witness = accepted_witness(
                raw, String(row.method), Int(row.index_a), index_b,
            )
            isnothing(witness) && continue
            epsilon_a = method in (:full, :full_TAC) ?
                Float64(row.epsilon_TotEmiss) :
                method == :full_ISI || method == :full_TotEmiss ?
                Float64(row.epsilon_TAC) :
                method == :group_TAC_TotEmiss ? Float64(row.epsilon_ISI) :
                method == :group_TAC_ISI ? Float64(row.epsilon_TotEmiss) :
                Float64(row.epsilon_TAC)
            epsilon_b = method in (:full, :full_TAC) ?
                Float64(row.epsilon_ISI) :
                method == :full_TotEmiss ? Float64(row.epsilon_ISI) :
                method == :full_ISI ? Float64(row.epsilon_TotEmiss) : nothing
            witness_start = row_start(witness, z_names)
            first = solve_scalarization(
                p, lower, ranges;
                mode=method, epsilon_a, epsilon_b, start=witness_start,
                time_limit=PARETO_SOLVE_TIME_LIMIT,
            )
            outcome = if first.accepted
                (; best=first, attempts=1, successful=1)
            else
                solve_with_deterministic_starts(
                    p, lower, ranges, anchors;
                    mode=method, epsilon_a, epsilon_b,
                    warm_starts=[witness_start],
                )
            end
            log[row_index, :attempts] += outcome.attempts
            log[row_index, :successful_starts] += outcome.successful
            if !isnothing(outcome.best)
                best = outcome.best
                new_row = result_row(
                    method, Int(row.index_a),
                    isnothing(index_b) ? missing : index_b,
                    row.epsilon_TAC, row.epsilon_TotEmiss, row.epsilon_ISI,
                    best, outcome.attempts, outcome.successful, z_names,
                )
                push!(raw, new_row; cols=:setequal, promote=true)
                log[row_index, :accepted] = true
                repaired += 1
                progress += 1
            end
            if hole_number % 20 == 0 || hole_number == length(holes)
                println(
                    two_dimensional ? "full" : "reduced",
                    " repair pass ", passes,
                    " | checked=", hole_number, "/", length(holes),
                    " | repaired_in_pass=", progress,
                )
            end
        end
        progress == 0 && break
    end
    unresolved = length(consistency_holes(log, raw, two_dimensional))
    return (; repaired, unresolved, passes)
end

function main_repair()
    p = pareto_parameters()
    anchors = anchor_data(p)
    payoff = reduce(vcat, [reshape([
        a.objectives.TAC, a.objectives.TotEmiss, a.objectives.ISI
    ], 1, 3) for a in anchors[1:3]])
    lower = [minimum(payoff[:, i]) for i in 1:3]
    upper = [maximum(payoff[:, i]) for i in 1:3]
    ranges = upper .- lower
    z_names = Symbol.("z" .* string.(eachindex(variable_names())))

    full = CSV.read(result_path("pareto_full_raw.csv"), DataFrame)
    reduced = CSV.read(result_path("pareto_reduced_raw.csv"), DataFrame)
    log = CSV.read(result_path("pareto_solve_log.csv"), DataFrame)
    full_log = log[startswith.(log.method, "full_"), :]
    reduced_log = log[startswith.(log.method, "group_"), :]

    full_result = repair_rows!(
        full, full_log, p, lower, ranges, anchors, z_names;
        two_dimensional=true,
    )
    reduced_result = repair_rows!(
        reduced, reduced_log, p, lower, ranges, anchors, z_names;
        two_dimensional=false,
    )

    combined_log = vcat(full_log, reduced_log; cols=:setequal)
    CSV.write(result_path("pareto_full_raw.csv"), full)
    CSV.write(result_path("pareto_reduced_raw.csv"), reduced)
    CSV.write(result_path("pareto_solve_log.csv"), combined_log)
    open(result_path("pareto_run.txt"), "a") do io
        println(io, "consistency_repair_at = ", Dates.now())
        println(io, "full_nested_holes_repaired = ", full_result.repaired)
        println(io, "full_nested_holes_unresolved = ", full_result.unresolved)
        println(io, "reduced_nested_holes_repaired = ", reduced_result.repaired)
        println(io, "reduced_nested_holes_unresolved = ", reduced_result.unresolved)
        println(io, "accepted_full_points_after_repair = ", nrow(full))
        println(io, "accepted_reduced_points_after_repair = ", nrow(reduced))
    end
    println("Full-grid repair: ", full_result)
    println("Reduced-front repair: ", reduced_result)
end

main_repair()
