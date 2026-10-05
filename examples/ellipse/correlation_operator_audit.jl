module CorrelationOperatorAudit

using LinearAlgebra
using Leiden
using Statistics

export current_code_correlation, paper_star_correlation, raw_gradient_cosine_correlation, reduced_groups, group_sets

unit(v; atol=1e-12) = norm(v) <= atol ? zero(v) : v / norm(v)

function project_to_row(v, a)
    denom = dot(a, a)
    denom <= 1e-12 && return copy(v)
    return v - dot(v, a) / denom * a
end

function logistic_weight(s; posmin=0.9, beta=100.0)
    return 1.0 - posmin * (1.0 / (1.0 + exp(-beta * s)))
end

function reduced_groups(A::AbstractMatrix, red::Integer)
    B = copy(Matrix{Float64}(A))
    for i in axes(B, 1), j in axes(B, 2)
        B[i, j] = round(B[i, j] * 1000, digits=0)
    end

    res = 1.0
    ccom = Leiden.leiden(B, resolution=res)
    part = ccom[2]
    guard = 0
    while length(part) != size(B, 1) && guard < 500
        ccom = Leiden.leiden(B, resolution=res)
        part = ccom[2]
        res += 1.0
        guard += 1
    end

    guard = 0
    while length(part) > red && res > 0 && guard < 500
        ccom = Leiden.leiden(B, resolution=res)
        part = ccom[2]
        res -= 0.1
        guard += 1
    end
    return part
end

function group_sets(part)
    if all(x -> x isa AbstractVector, part)
        return sort([sort(collect(Int, v)) for v in part], by=x -> first(x))
    end

    groups = Dict{Any, Vector{Int}}()
    for (idx, label) in enumerate(part)
        push!(get!(groups, label, Int[]), idx)
    end
    return sort([sort(v) for v in values(groups)], by=x -> first(x))
end

function normalized_objective_rows(ObjMatrix)
    normc = zeros(size(ObjMatrix))
    for i in axes(ObjMatrix, 1)
        normc[i, :] = -unit(vec(ObjMatrix[i, :]))
    end
    return normc
end

function current_code_correlation(IneqMatrix, ObjMatrix, EqMatrix, NoOfSP; posmin=0.9, beta=100.0)
    NoOfObj = Int(size(ObjMatrix, 1) / NoOfSP)
    normc = normalized_objective_rows(ObjMatrix)

    total_weight = zeros(NoOfObj, NoOfObj)
    total_strength_weight = zeros(NoOfObj, NoOfObj)

    for i in 1:NoOfObj, j in i+1:NoOfObj
        for k in axes(IneqMatrix, 1), n in 1:NoOfSP
            row_i = (i - 1) * NoOfSP + n
            row_j = (j - 1) * NoOfSP + n
            a = vec(IneqMatrix[k, :])
            vi = vec(normc[row_i, :])
            vj = vec(normc[row_j, :])
            pi = project_to_row(vi, a)
            pj = project_to_row(vj, a)
            si = dot(vi, a)
            sj = dot(vj, a)

            use_pair = true
            if abs(si) > 1e-12 && abs(sj) > 1e-12
                use_pair = si > 0.0 && sj > 0.0
            end
            if use_pair
                s = dot(unit(pi), unit(pj))
                w = logistic_weight(s; posmin=posmin, beta=beta)
                total_weight[i, j] += w
                total_strength_weight[i, j] += w * s
            end
        end

        if sum(EqMatrix) != 0
            for k in axes(EqMatrix, 1), n in 1:NoOfSP
                row_i = (i - 1) * NoOfSP + n
                row_j = (j - 1) * NoOfSP + n
                a = vec(EqMatrix[k, :])
                vi = vec(normc[row_i, :])
                vj = vec(normc[row_j, :])
                s = dot(unit(project_to_row(vi, a)), unit(project_to_row(vj, a)))
                total_weight[i, j] += 1.0
                total_strength_weight[i, j] += s
            end
        end

        total_weight[j, i] = total_weight[i, j]
        total_strength_weight[j, i] = total_strength_weight[i, j]
    end

    A = Matrix{Float64}(I, NoOfObj, NoOfObj)
    for i in 1:NoOfObj, j in i+1:NoOfObj
        A[i, j] = A[j, i] = total_weight[i, j] <= 1e-12 ? 0.0 : 0.5 * (1.0 + total_strength_weight[i, j] / total_weight[i, j])
    end
    return A, total_weight, total_strength_weight
end

function paper_star_correlation(IneqMatrix, ObjMatrix, EqMatrix, NoOfSP; posmin=0.9, beta=100.0)
    NoOfObj = Int(size(ObjMatrix, 1) / NoOfSP)
    normc = normalized_objective_rows(ObjMatrix)

    total_weight = zeros(NoOfObj, NoOfObj)
    total_strength_weight = zeros(NoOfObj, NoOfObj)

    for i in 1:NoOfObj, j in i+1:NoOfObj
        for k in axes(IneqMatrix, 1), n in 1:NoOfSP
            row_i = (i - 1) * NoOfSP + n
            row_j = (j - 1) * NoOfSP + n
            a = vec(IneqMatrix[k, :])
            vi = vec(normc[row_i, :])
            vj = vec(normc[row_j, :])

            di = dot(vi, a) > 0.0 ? unit(project_to_row(vi, a)) : vi
            dj = dot(vj, a) > 0.0 ? unit(project_to_row(vj, a)) : vj
            s = dot(unit(di), unit(dj))
            w = logistic_weight(s; posmin=posmin, beta=beta)
            total_weight[i, j] += w
            total_strength_weight[i, j] += w * s
        end

        if sum(EqMatrix) != 0
            for k in axes(EqMatrix, 1), n in 1:NoOfSP
                row_i = (i - 1) * NoOfSP + n
                row_j = (j - 1) * NoOfSP + n
                a = vec(EqMatrix[k, :])
                vi = vec(normc[row_i, :])
                vj = vec(normc[row_j, :])
                s = dot(unit(project_to_row(vi, a)), unit(project_to_row(vj, a)))
                total_weight[i, j] += 1.0
                total_strength_weight[i, j] += s
            end
        end

        total_weight[j, i] = total_weight[i, j]
        total_strength_weight[j, i] = total_strength_weight[i, j]
    end

    A = Matrix{Float64}(I, NoOfObj, NoOfObj)
    for i in 1:NoOfObj, j in i+1:NoOfObj
        A[i, j] = A[j, i] = total_weight[i, j] <= 1e-12 ? 0.0 : 0.5 * (1.0 + total_strength_weight[i, j] / total_weight[i, j])
    end
    return A, total_weight, total_strength_weight
end

function raw_gradient_cosine_correlation(ObjMatrix, NoOfSP)
    NoOfObj = Int(size(ObjMatrix, 1) / NoOfSP)
    normc = normalized_objective_rows(ObjMatrix)
    A = Matrix{Float64}(I, NoOfObj, NoOfObj)

    for i in 1:NoOfObj, j in i+1:NoOfObj
        sims = Float64[]
        for n in 1:NoOfSP
            row_i = (i - 1) * NoOfSP + n
            row_j = (j - 1) * NoOfSP + n
            vi = vec(normc[row_i, :])
            vj = vec(normc[row_j, :])
            if norm(vi) > 1e-12 && norm(vj) > 1e-12
                push!(sims, dot(vi, vj))
            end
        end
        mean_sim = isempty(sims) ? 0.0 : mean(sims)
        A[i, j] = A[j, i] = clamp(0.5 * (1.0 + mean_sim), 0.0, 1.0)
    end
    return A
end

end
