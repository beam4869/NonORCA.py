using CSV
using DataFrames
using ForwardDiff
using Ipopt
using JuMP
using LinearAlgebra
using Random

include("correlation_operator_audit.jl")
using .CorrelationOperatorAudit

const M = 5
const I_DTLZ = 3
const K_TAIL = 10
const N = M + K_TAIL - 1
const ITERA = 15
const NPART = M - I_DTLZ + 1
const OUTDIR = @__DIR__

Random.seed!(1234)

gfun(x) = sum((x[i] - 0.5)^2 for i in 1:N)
theta3(x) = pi * (1 + 2 * gfun(x) * x[3]) / (4 * (1 + gfun(x)))
theta4(x) = pi * (1 + 2 * gfun(x) * x[4]) / (4 * (1 + gfun(x)))

ff1(x) = (1 + gfun(x)) * cos(pi * x[1] / 2) * cos(pi * x[2] / 2) * cos(theta3(x)) * cos(theta4(x))
ff2(x) = (1 + gfun(x)) * cos(pi * x[1] / 2) * cos(pi * x[2] / 2) * cos(theta3(x)) * sin(theta4(x))
ff3(x) = (1 + gfun(x)) * cos(pi * x[1] / 2) * cos(pi * x[2] / 2) * sin(theta3(x))
ff4(x) = (1 + gfun(x)) * cos(pi * x[1] / 2) * sin(pi * x[2] / 2)
ff5(x) = (1 + gfun(x)) * sin(pi * x[1] / 2)

const OBJFUNCS = [ff1, ff2, ff3, ff4, ff5]
const OBJLABELS = ["f1", "f2", "f3", "f4", "f5"]

function solve_vertices()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x[1:N] <= 1, start = 0.5)
    @NLexpression(model, gx, sum((x[i] - 0.5)^2 for i in 1:N))
    @NLexpression(model, f1, (1 + gx) * cos(pi * x[1] / 2) * cos(pi * x[2] / 2) *
                             cos(pi * (1 + 2 * gx * x[3]) / (4 * (1 + gx))) *
                             cos(pi * (1 + 2 * gx * x[4]) / (4 * (1 + gx))))
    @NLexpression(model, f2, (1 + gx) * cos(pi * x[1] / 2) * cos(pi * x[2] / 2) *
                             cos(pi * (1 + 2 * gx * x[3]) / (4 * (1 + gx))) *
                             sin(pi * (1 + 2 * gx * x[4]) / (4 * (1 + gx))))
    @NLexpression(model, f3, (1 + gx) * cos(pi * x[1] / 2) * cos(pi * x[2] / 2) *
                             sin(pi * (1 + 2 * gx * x[3]) / (4 * (1 + gx))))
    @NLexpression(model, f4, (1 + gx) * cos(pi * x[1] / 2) * sin(pi * x[2] / 2))
    @NLexpression(model, f5, (1 + gx) * sin(pi * x[1] / 2))
    objexprs = [f1, f2, f3, f4, f5]

    vertices = Vector{Vector{Float64}}()
    for i in 1:M
        @NLobjective(model, Min, objexprs[i])
        optimize!(model)
        push!(vertices, value.(x))
    end
    return vertices
end

function bound_ineq_matrix()
    eye = Matrix{Float64}(I, N, N)
    return vcat(eye, eye)
end

function objective_matrix(objfuncs, selected_points)
    C = zeros(length(objfuncs) * length(selected_points), N)
    row = 1
    for f in objfuncs
        for sp in selected_points
            C[row, :] = ForwardDiff.gradient(f, sp)
            row += 1
        end
    end
    return C
end

function project_to_bounds(x)
    return clamp.(x, 0.0, 1.0)
end

function safe_normalize(v)
    nv = norm(v)
    return nv <= 1e-12 ? zero(v) : v / nv
end

function project_to_row(v, a)
    denom = dot(a, a)
    return denom <= 1e-12 ? copy(v) : v - dot(v, a) / denom * a
end

function delta_generation(Ae, ce, de)
    normc = zeros(size(ce))
    for i in axes(ce, 1)
        normc[i, :] = -safe_normalize(vec(ce[i, :]))
    end
    omega = rand(size(ce, 1))
    denom = sum(omega)

    if sum(de) != 0
        accum = zeros(size(ce, 2))
        for k in axes(Ae, 1)
            accum += sum(omega[i] * project_to_row(vec(normc[i, :]), vec(Ae[k, :])) for i in axes(ce, 1)) / denom
        end
        for k in axes(de, 1)
            accum += sum(omega[i] * project_to_row(vec(normc[i, :]), vec(de[k, :])) for i in axes(ce, 1)) / denom
        end
        return accum / (size(Ae, 1) + size(de, 1))
    end

    accum = zeros(size(ce, 2))
    for k in axes(Ae, 1)
        accum += sum(omega[i] * project_to_row(vec(normc[i, :]), vec(Ae[k, :])) for i in axes(ce, 1)) / denom
    end
    return accum / size(Ae, 1)
end

function generate_selected_points(vertices, Amats, Dmats)
    selected_points = Vector{Vector{Float64}}()
    for vertex in vertices
        push!(selected_points, vertex)
        for _ in 1:ITERA
            current = selected_points[end]
            Cstep = objective_matrix(OBJFUNCS, [current])
            delta = delta_generation(Amats, Cstep, Dmats)
            push!(selected_points, project_to_bounds(current + delta))
        end
    end
    return selected_points
end

function matrix_dataframe(A)
    df = DataFrame(objective = OBJLABELS)
    for (j, label) in enumerate(OBJLABELS)
        df[!, Symbol(label)] = A[:, j]
    end
    return df
end

function write_selected_points(path, selected_points)
    df = DataFrame(point_id = collect(1:length(selected_points)))
    for j in 1:N
        df[!, Symbol("x$j")] = [p[j] for p in selected_points]
    end
    CSV.write(path, df)
end

function run()
    vertices = solve_vertices()
    Amats = bound_ineq_matrix()
    Dmats = zeros(1, N)
    selected_points = generate_selected_points(vertices, Amats, Dmats)
    Cmat = objective_matrix(OBJFUNCS, selected_points)

    A_current, W_current, SW_current = current_code_correlation(Amats, Cmat, Dmats, length(selected_points))
    A_star, W_star, SW_star = paper_star_correlation(Amats, Cmat, Dmats, length(selected_points))

    part_current = reduced_groups(A_current, NPART)
    part_star = reduced_groups(A_star, NPART)
    sets_current = group_sets(part_current)
    sets_star = group_sets(part_star)

    CSV.write(joinpath(OUTDIR, "audit_dtlz5_current_matrix.csv"), matrix_dataframe(A_current))
    CSV.write(joinpath(OUTDIR, "audit_dtlz5_paper_star_matrix.csv"), matrix_dataframe(A_star))
    CSV.write(joinpath(OUTDIR, "audit_dtlz5_current_total_weight.csv"), matrix_dataframe(W_current))
    CSV.write(joinpath(OUTDIR, "audit_dtlz5_paper_star_total_weight.csv"), matrix_dataframe(W_star))
    CSV.write(joinpath(OUTDIR, "audit_dtlz5_current_strength_weight.csv"), matrix_dataframe(SW_current))
    CSV.write(joinpath(OUTDIR, "audit_dtlz5_paper_star_strength_weight.csv"), matrix_dataframe(SW_star))
    write_selected_points(joinpath(OUTDIR, "audit_dtlz5_selected_points.csv"), selected_points)

    summary = DataFrame(
        case = ["DTLZ5(3,5)", "DTLZ5(3,5)"],
        operator = ["current-code", "paper-star"],
        selected_points = [length(selected_points), length(selected_points)],
        requested_groups = [NPART, NPART],
        partition = [string(part_current), string(part_star)],
        group_sets = [string(sets_current), string(sets_star)],
        expected_group_sets = ["[[1, 2, 3], [4], [5]]", "[[1, 2, 3], [4], [5]]"],
    )
    CSV.write(joinpath(OUTDIR, "audit_dtlz5_current_vs_paper_star_summary.csv"), summary)

    println(summary)
    println("current-code matrix:")
    show(stdout, "text/plain", A_current)
    println("\n\npaper-star matrix:")
    show(stdout, "text/plain", A_star)
    println("\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
