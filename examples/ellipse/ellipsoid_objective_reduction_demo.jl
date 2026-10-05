using LinearAlgebra
using Random
using DelimitedFiles
using Printf

# Simple nonlinear feasible set:
#     x'Qx <= 1
# where Q = diag(1/a^2, 1/b^2, 1/c^2).
#
# Linear objectives, all minimized:
#     f1(x) = -x1
#     f2(x) = -0.99x1 - 0.05x2
#     f3(x) =  x1
#     f4(x) = -x3
#
# Expected structure:
#     f1 and f2 are deliberately near-collinear, f3 conflicts
#     with f1/f2, and f4 is an additional orthogonal direction.

const AXES = [1.40, 0.90, 0.60]
const Q = Diagonal(1.0 ./ (AXES .^ 2))
const C = [
    -1.00   0.00   0.00
    -0.99  -0.05   0.00
     1.00   0.00   0.00
     0.00   0.00  -1.00
]
const OBJ_NAMES = ["f1 = -x1", "f2 = -0.99x1 - 0.05x2", "f3 = x1", "f4 = -x3"]

unit(v; atol=1e-12) = norm(v) <= atol ? zero(v) : v / norm(v)
ellipsoid_value(x, Qmat=Q) = dot(x, Qmat * x) - 1.0
ellipsoid_grad(x, Qmat=Q) = 2.0 .* (Qmat * x)
objective_value(c, x) = dot(c, x)

function single_objective_optimum(c, Qmat=Q)
    # Closed-form minimizer of c'x subject to x'Qx <= 1.
    qinv_c = Qmat \ c
    denom = sqrt(dot(c, qinv_c))
    return -qinv_c / denom
end

function project_to_ellipsoid(y, Qmat=Q; tol=1e-12, max_iter=200)
    # Euclidean projection of y onto {x: x'Qx <= 1}.
    if ellipsoid_value(y, Qmat) <= tol
        return copy(y)
    end

    q = diag(Qmat)
    function residual(lambda)
        return sum(q[i] * y[i]^2 / (1.0 + lambda * q[i])^2 for i in eachindex(y)) - 1.0
    end

    lo, hi = 0.0, 1.0
    while residual(hi) > 0.0
        hi *= 2.0
    end

    for _ in 1:max_iter
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0
            lo = mid
        else
            hi = mid
        end
        if hi - lo <= tol * max(1.0, hi)
            break
        end
    end

    lambda = 0.5 * (lo + hi)
    return y ./ (1.0 .+ lambda .* q)
end

function tangent_projection(v, x, Qmat=Q)
    n = ellipsoid_grad(x, Qmat)
    return v - dot(v, n) / dot(n, n) * n
end

function projected_direction_data(x, Cmat=C, Qmat=Q)
    # For a boundary point x, the paper's selected-point direction is built
    # from objective descent directions projected onto the local tangent plane.
    normal = unit(ellipsoid_grad(x, Qmat))
    raw_dirs = [unit(-Cmat[i, :]) for i in axes(Cmat, 1)]
    projected_dirs = [tangent_projection(raw_dirs[i], x, Qmat) for i in eachindex(raw_dirs)]
    projected_norms = [norm(projected_dirs[i]) for i in eachindex(projected_dirs)]
    unit_projected_dirs = [unit(projected_dirs[i]) for i in eachindex(projected_dirs)]
    normal_dots = [dot(raw_dirs[i], normal) for i in eachindex(raw_dirs)]
    return raw_dirs, projected_dirs, projected_norms, unit_projected_dirs, normal_dots
end

function step_direction(x, Cmat=C, Qmat=Q)
    # Equation-like analogue of the paper's random conic combination of
    # projected objective gradients. We use descent directions -grad f_i.
    raw_dirs, projected_dirs, projected_norms, directions, normal_dots =
        projected_direction_data(x, Cmat, Qmat)
    weights = rand(length(directions))
    d = sum(weights[i] .* projected_dirs[i] for i in eachindex(projected_dirs)) / sum(weights)
    return d, weights, raw_dirs, projected_dirs, projected_norms, directions, normal_dots
end

function generate_selected_points(; alpha=0.35, steps_per_seed=18, seed=12)
    Random.seed!(seed)
    vertices = [single_objective_optimum(C[i, :]) for i in axes(C, 1)]

    points = Vector{Vector{Float64}}()
    point_meta = Vector{Tuple{Int, Int}}()
    step_records = Vector{Vector{Any}}()
    direction_records = Vector{Vector{Any}}()
    for seed_idx in eachindex(vertices)
        x = vertices[seed_idx]
        push!(points, copy(x))
        push!(point_meta, (seed_idx, 0))
        for step in 1:steps_per_seed
            d, weights, raw_dirs, projected_dirs, projected_norms, unit_projected_dirs, normal_dots =
                step_direction(x)
            trial = x + alpha * d
            next_x = project_to_ellipsoid(trial)

            push!(step_records, Any[
                seed_idx, step,
                x[1], x[2], x[3], ellipsoid_value(x),
                d[1], d[2], d[3],
                trial[1], trial[2], trial[3], ellipsoid_value(trial),
                next_x[1], next_x[2], next_x[3], ellipsoid_value(next_x),
                norm(next_x - trial)
            ])

            for obj_idx in axes(C, 1)
                push!(direction_records, Any[
                    seed_idx, step, obj_idx, weights[obj_idx],
                    raw_dirs[obj_idx][1], raw_dirs[obj_idx][2], raw_dirs[obj_idx][3],
                    normal_dots[obj_idx],
                    projected_dirs[obj_idx][1], projected_dirs[obj_idx][2], projected_dirs[obj_idx][3],
                    projected_norms[obj_idx],
                    unit_projected_dirs[obj_idx][1], unit_projected_dirs[obj_idx][2], unit_projected_dirs[obj_idx][3]
                ])
            end

            x = next_x
            push!(points, copy(x))
            push!(point_meta, (seed_idx, step))
        end
    end

    return vertices, points, point_meta, step_records, direction_records
end

function projected_pair_strength(ci, cj, x, Qmat=Q; normal_tol=1e-12)
    # This mirrors the projected-vector branch in NLPCorrStrengGenerating:
    # normalize objective descent directions, project them onto the local
    # linearized constraint surface, and only retain inequality-constraint
    # contributions when both directions point outward relative to that row.
    vi = unit(-ci)
    vj = unit(-cj)
    normal = ellipsoid_grad(x, Qmat)
    pi = tangent_projection(vi, x, Qmat)
    pj = tangent_projection(vj, x, Qmat)
    si = dot(vi, normal)
    sj = dot(vj, normal)

    use_pair = true
    if abs(si) > normal_tol && abs(sj) > normal_tol
        use_pair = si > 0.0 && sj > 0.0
    end

    if !use_pair
        return 0.0, 0.0, si, sj
    end
    return dot(unit(pi), unit(pj)), 1.0, si, sj
end

function objective_correlation(points, Cmat=C, Qmat=Q; posmin=0.9, beta=100.0)
    m = size(Cmat, 1)
    A = Matrix{Float64}(I, m, m)
    raw_mean = Matrix{Float64}(I, m, m)
    total_weight = zeros(m, m)

    for i in 1:m
        for j in i+1:m
            weighted_sum = 0.0
            weight_total = 0.0
            raw_sum = 0.0
            for x in points
                s, active, _, _ = projected_pair_strength(Cmat[i, :], Cmat[j, :], x, Qmat)
                w = active * (1.0 - posmin * (1.0 / (1.0 + exp(-beta * s))))
                weighted_sum += w * s
                weight_total += w
                raw_sum += s
            end
            score = weight_total <= 1e-12 ? 0.0 : 0.5 * (1.0 + weighted_sum / weight_total)
            A[i, j] = A[j, i] = score
            raw_mean[i, j] = raw_mean[j, i] = raw_sum / length(points)
            total_weight[i, j] = total_weight[j, i] = weight_total
        end
    end

    for i in 1:m
        total_weight[i, i] = 1.0
    end

    return A, raw_mean, total_weight
end

function unprojected_objective_matrix(Cmat=C)
    m = size(Cmat, 1)
    A = Matrix{Float64}(I, m, m)
    dirs = [unit(-Cmat[i, :]) for i in axes(Cmat, 1)]
    for i in 1:m
        for j in i+1:m
            score = 0.5 * (1.0 + dot(dirs[i], dirs[j]))
            A[i, j] = A[j, i] = score
        end
    end
    return A
end

function connected_components_from_threshold(A; threshold=0.68)
    m = size(A, 1)
    seen = falses(m)
    groups = Vector{Vector{Int}}()
    for start in 1:m
        seen[start] && continue
        stack = [start]
        seen[start] = true
        group = Int[]
        while !isempty(stack)
            node = pop!(stack)
            push!(group, node)
            for nbr in 1:m
                if !seen[nbr] && nbr != node && A[node, nbr] >= threshold
                    seen[nbr] = true
                    push!(stack, nbr)
                end
            end
        end
        push!(groups, sort(group))
    end
    return groups
end

function write_outputs(vertices, points, point_meta, step_records, direction_records, A, raw_mean, total_weight, raw_objective, groups)
    outdir = @__DIR__

    point_rows = Array{Any}(undef, length(points) + 1, 7)
    point_rows[1, :] = ["point_id", "seed_objective", "step", "x1", "x2", "x3", "g(x)"]
    for (idx, x) in enumerate(points)
        seed_idx, step = point_meta[idx]
        point_rows[idx + 1, :] = [idx, seed_idx, step, x[1], x[2], x[3], ellipsoid_value(x)]
    end
    writedlm(joinpath(outdir, "ellipsoid_selected_points.csv"), point_rows, ',')

    step_header = [
        "seed_objective", "step",
        "current_x1", "current_x2", "current_x3", "current_g",
        "step_direction_x1", "step_direction_x2", "step_direction_x3",
        "trial_x1", "trial_x2", "trial_x3", "trial_g",
        "projected_x1", "projected_x2", "projected_x3", "projected_g",
        "projection_distance"
    ]
    step_rows = Array{Any}(undef, length(step_records) + 1, length(step_header))
    step_rows[1, :] = step_header
    for (idx, row) in enumerate(step_records)
        step_rows[idx + 1, :] = row
    end
    writedlm(joinpath(outdir, "ellipsoid_step_details.csv"), step_rows, ',')

    direction_header = [
        "seed_objective", "step", "objective", "random_weight",
        "descent_x1", "descent_x2", "descent_x3", "normal_dot_descent",
        "projected_x1", "projected_x2", "projected_x3", "projected_norm",
        "unit_projected_x1", "unit_projected_x2", "unit_projected_x3"
    ]
    direction_rows = Array{Any}(undef, length(direction_records) + 1, length(direction_header))
    direction_rows[1, :] = direction_header
    for (idx, row) in enumerate(direction_records)
        direction_rows[idx + 1, :] = row
    end
    writedlm(joinpath(outdir, "ellipsoid_projected_directions.csv"), direction_rows, ',')

    vertex_rows = Array{Any}(undef, length(vertices) + 1, 8)
    vertex_rows[1, :] = ["objective", "x1", "x2", "x3", "f1", "f2", "f3", "f4"]
    for (idx, x) in enumerate(vertices)
        vertex_rows[idx + 1, :] = vcat([OBJ_NAMES[idx], x[1], x[2], x[3]], [objective_value(C[i, :], x) for i in axes(C, 1)])
    end
    writedlm(joinpath(outdir, "ellipsoid_objective_vertices.csv"), vertex_rows, ',')

    header = reshape(vcat(["objective"], ["f$i" for i in axes(C, 1)]), 1, :)
    score_rows = vcat(header, hcat(["f$i" for i in axes(C, 1)], A))
    writedlm(joinpath(outdir, "ellipsoid_correlation_matrix.csv"), score_rows, ',')

    raw_rows = vcat(header, hcat(["f$i" for i in axes(C, 1)], raw_mean))
    writedlm(joinpath(outdir, "ellipsoid_raw_alignment_matrix.csv"), raw_rows, ',')

    weight_rows = vcat(header, hcat(["f$i" for i in axes(C, 1)], total_weight))
    writedlm(joinpath(outdir, "ellipsoid_total_weight_matrix.csv"), weight_rows, ',')

    raw_objective_rows = vcat(header, hcat(["f$i" for i in axes(C, 1)], raw_objective))
    writedlm(joinpath(outdir, "ellipsoid_unprojected_objective_matrix.csv"), raw_objective_rows, ',')

    open(joinpath(outdir, "ellipsoid_groups.txt"), "w") do io
        println(io, "Threshold grouping on objective correlation matrix")
        println(io, "threshold = 0.68")
        for (idx, group) in enumerate(groups)
            labels = join(["f$i" for i in group], ", ")
            println(io, "G$idx = {$labels}")
        end
    end
end

function main()
    vertices, points, point_meta, step_records, direction_records = generate_selected_points()
    A, raw_mean, total_weight = objective_correlation(points)
    raw_objective = unprojected_objective_matrix()
    groups = connected_components_from_threshold(A)
    write_outputs(vertices, points, point_meta, step_records, direction_records, A, raw_mean, total_weight, raw_objective, groups)

    println("Ellipsoid axes: ", AXES)
    println("\nObjective seed points:")
    for (idx, x) in enumerate(vertices)
        @printf("  f%d*: [% .4f, % .4f, % .4f]\n", idx, x[1], x[2], x[3])
    end

    println("\nObjective correlation matrix A:")
    for i in axes(A, 1)
        println("  ", join([@sprintf("%0.3f", A[i, j]) for j in axes(A, 2)], "  "))
    end

    println("\nRaw projected-gradient alignment:")
    for i in axes(raw_mean, 1)
        println("  ", join([@sprintf("% .3f", raw_mean[i, j]) for j in axes(raw_mean, 2)], "  "))
    end

    println("\nUnprojected objective angle matrix:")
    for i in axes(raw_objective, 1)
        println("  ", join([@sprintf("%0.3f", raw_objective[i, j]) for j in axes(raw_objective, 2)], "  "))
    end

    println("\nTotal retained weights:")
    for i in axes(total_weight, 1)
        println("  ", join([@sprintf("%0.3f", total_weight[i, j]) for j in axes(total_weight, 2)], "  "))
    end

    println("\nGroups:")
    for (idx, group) in enumerate(groups)
        labels = join(["f$i" for i in group], ", ")
        println("  G$idx = {$labels}")
    end

    println("\nAudit outputs:")
    println("  ellipsoid_step_details.csv")
    println("  ellipsoid_projected_directions.csv")
end

main()
