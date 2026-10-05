using LinearAlgebra
using Random
using DelimitedFiles
using Printf

const AXES = [1.40, 0.90, 0.60]
const Q = Diagonal(1.0 ./ (AXES .^ 2))
const BASE_C = [
    -1.00   0.00   0.00
    -0.99  -0.05   0.00
     1.00   0.00   0.00
     0.00   0.00  -1.00
]

unit(v; atol=1e-12) = norm(v) <= atol ? zero(v) : v / norm(v)
ellipsoid_value(x, Qmat=Q) = dot(x, Qmat * x) - 1.0
ellipsoid_grad(x, Qmat=Q) = 2.0 .* (Qmat * x)

function single_objective_optimum(c, Qmat=Q)
    qinv_c = Qmat \ c
    denom = sqrt(dot(c, qinv_c))
    return -qinv_c / denom
end

function project_to_ellipsoid(y, Qmat=Q; tol=1e-12, max_iter=200)
    if ellipsoid_value(y, Qmat) <= tol
        return copy(y)
    end
    q = diag(Qmat)
    residual(lambda) = sum(q[i] * y[i]^2 / (1.0 + lambda * q[i])^2 for i in eachindex(y)) - 1.0
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

function build_objectives(extra_near_f3; eps=0.04, include_f2=true)
    rows = include_f2 ? [BASE_C[i, :] for i in axes(BASE_C, 1)] : [BASE_C[1, :], BASE_C[3, :], BASE_C[4, :]]
    for k in 1:extra_near_f3
        theta = 2.0 * pi * (k - 1) / max(extra_near_f3, 1)
        push!(rows, [1.0, eps * cos(theta), eps * sin(theta)])
    end
    return reduce(vcat, reshape.(rows, 1, 3))
end

function generate_selected_points(Cmat; alpha=0.35, steps_per_seed=18, seed=12)
    Random.seed!(seed)
    vertices = [single_objective_optimum(Cmat[i, :]) for i in axes(Cmat, 1)]
    points = Vector{Vector{Float64}}()
    for x0 in vertices
        x = copy(x0)
        push!(points, copy(x))
        for _ in 1:steps_per_seed
            raw_dirs = [unit(-Cmat[i, :]) for i in axes(Cmat, 1)]
            projected_dirs = [tangent_projection(raw_dirs[i], x) for i in eachindex(raw_dirs)]
            weights = rand(length(projected_dirs))
            d = sum(weights[i] .* projected_dirs[i] for i in eachindex(projected_dirs)) / sum(weights)
            x = project_to_ellipsoid(x + alpha * d)
            push!(points, copy(x))
        end
    end
    return points
end

function code_gate_score(points, ci, cj; posmin=0.9, beta=100.0)
    weighted_sum = 0.0
    weight_total = 0.0
    active_count = 0
    for x in points
        vi = unit(-ci)
        vj = unit(-cj)
        normal = ellipsoid_grad(x)
        pi = tangent_projection(vi, x)
        pj = tangent_projection(vj, x)
        si = dot(vi, normal)
        sj = dot(vj, normal)
        use_pair = true
        if abs(si) > 1e-12 && abs(sj) > 1e-12
            use_pair = si > 0.0 && sj > 0.0
        end
        if use_pair
            s = dot(unit(pi), unit(pj))
            w = 1.0 - posmin * (1.0 / (1.0 + exp(-beta * s)))
            weighted_sum += w * s
            weight_total += w
            active_count += 1
        end
    end
    score = weight_total <= 1e-12 ? 0.0 : 0.5 * (1.0 + weighted_sum / weight_total)
    return score, weight_total, active_count
end

function paper_star_direction(c, x)
    v = unit(-c)
    normal = ellipsoid_grad(x)
    if dot(v, normal) > 0.0
        return unit(tangent_projection(v, x))
    end
    return v
end

function paper_star_score(points, ci, cj; posmin=0.9, beta=100.0)
    weighted_sum = 0.0
    weight_total = 0.0
    for x in points
        vi = paper_star_direction(ci, x)
        vj = paper_star_direction(cj, x)
        s = dot(vi, vj)
        w = 1.0 - posmin * (1.0 / (1.0 + exp(-beta * s)))
        weighted_sum += w * s
        weight_total += w
    end
    return 0.5 * (1.0 + weighted_sum / weight_total), weight_total
end

function summarize_points(points)
    left = count(x -> x[1] < 0.0, points) / length(points)
    left_top = count(x -> x[1] < 0.0 && x[3] > 0.0, points) / length(points)
    right_top = count(x -> x[1] > 0.0 && x[3] > 0.0, points) / length(points)
    return left, left_top, right_top
end

function main()
    extras = [0, 1, 2, 5, 10, 20, 40]
    rows = Any[
        [
            "include_f2", "near_f3_objectives", "total_objectives", "selected_points",
            "left_fraction", "left_top_fraction", "right_top_fraction",
            "f1_f4_code_gate", "f1_f4_code_weight", "f1_f4_code_active_points",
            "f1_f4_paper_star", "f1_f4_paper_weight",
            "f1_f2_code_gate", "f1_f3_code_gate"
        ]
    ]

    println("include_f2  near_f3  total_obj  selected  left   left_top  right_top  f1-f4(code)  f1-f4(star)")
    for include_f2 in (true, false)
        for extra in extras
            Cmat = build_objectives(extra; include_f2=include_f2)
            points = generate_selected_points(Cmat)
            left, left_top, right_top = summarize_points(points)
            f3_idx = include_f2 ? 3 : 2
            f4_idx = include_f2 ? 4 : 3
            f14_code, f14_w, f14_active = code_gate_score(points, Cmat[1, :], Cmat[f4_idx, :])
            f14_star, f14_star_w = paper_star_score(points, Cmat[1, :], Cmat[f4_idx, :])
            f12_code = include_f2 ? code_gate_score(points, Cmat[1, :], Cmat[2, :])[1] : NaN
            f13_code, _, _ = code_gate_score(points, Cmat[1, :], Cmat[f3_idx, :])
            @printf("%10s  %7d  %9d  %8d  %5.2f  %8.2f  %9.2f  %11.3f  %11.3f\n",
                string(include_f2), extra, size(Cmat, 1), length(points), left, left_top, right_top, f14_code, f14_star)
            push!(rows, Any[
                include_f2, extra, size(Cmat, 1), length(points),
                left, left_top, right_top,
                f14_code, f14_w, f14_active,
                f14_star, f14_star_w,
                f12_code, f13_code
            ])
        end
    end
    writedlm(joinpath(@__DIR__, "ellipsoid_near_f3_sensitivity.csv"), rows, ',')
end

main()
