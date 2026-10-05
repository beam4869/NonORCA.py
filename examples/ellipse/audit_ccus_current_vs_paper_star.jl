using CPLEX
using CSV
using DataFrames
using ForwardDiff
using Ipopt
using JuMP
using LinearAlgebra
using NLPModels
using NLPModelsJuMP
using Random

import MathOptInterface as MOI

include("correlation_operator_audit.jl")
using .CorrelationOperatorAudit

const OUTDIR = @__DIR__
const CCUS_ITERA = parse(Int, get(ENV, "CCUS_AUDIT_ITERA", "20"))
const STEP_SCALE = parse(Float64, get(ENV, "CCUS_AUDIT_STEP_SCALE", "1.0"))
const SEED_TIME_LIMIT = parse(Float64, get(ENV, "CCUS_AUDIT_SEED_TIME", "60.0"))
const APPEND_ZERO_SEEDS = lowercase(get(ENV, "CCUS_AUDIT_APPEND_ZERO_SEEDS", "true")) == "true"
const FINAL_INEQ_MODE = get(ENV, "CCUS_AUDIT_FINAL_INEQ", "bounds")

Random.seed!(1234)

const S = 4
const K = 6
const T = 1

const M_s = [357.0, 1260.0, 399.0, 3426.0]
const M_s_TOTAL = sum(M_s)
const L_s = zeros(Float64, S)
const y_s = [1.0, 0.44, 0.27, 0.07]
const CO2_gen = sum(M_s .* y_s)
const y_s_t = reshape([1.0, 0.999, 0.999, 0.999], S, T)

const Zmin_k = [0.06, 0.94, 0.94, 0.999, 0.999, 0.999]
const Gmax_k = [126.0, 508.0, 3168.0, 756.0, 543.0, 913.0]
const Mu_k = [0.28, 0.5, 0.0, 0.17, 0.11, 0.34]

const ComprePowPara_s_k = [0.43 2.22 4.37 4.37 4.37 4.37;
                           0.49 2.23 4.37 4.37 4.37 4.37;
                           0.62 2.28 4.37 4.37 4.37 4.37;
                           0.58 2.28 4.37 4.37 4.37 4.37] .|> Float64

const PumpPowPara_s_k = [0.0 0.0 0.204 0.017 0.176 0.204;
                         0.0 0.0 0.204 0.017 0.176 0.204;
                         0.0 0.0 0.205 0.018 0.176 0.203;
                         0.0 0.0 0.205 0.018 0.175 0.203] .|> Float64

const PressDropPara_s_k = [56.39 832.13 49.51 49.51 50.82 51.15;
                           67.87 843.61 60.98 60.98 62.30 62.62;
                           90.82 896.07 96.72 96.72 29.84 16.72;
                           82.95 888.20 88.85 88.85 21.64 26.89] .|> Float64

const DistSourSink_s_k = [1.72 25.38 1.51 1.51 1.55 1.56;
                          2.07 25.73 1.86 1.86 1.90 1.91;
                          2.77 27.33 2.95 2.95 0.91 0.51;
                          2.53 27.09 2.71 2.71 0.66 0.82] .|> Float64

const epsilon_t = [0.9]
const epsilon_p = 0.366
const powerprice = 0.02
const CRF = 0.15
const gamma_t = [0.0338]
const CapturePercent = 0.4
const TranspCost_s_t = reshape([0.0, 29.0, 43.0, 35.0], S, T)

const I_tot_r = [12.0]
const I_tot_t = [19.0]
const I_tot_k = [7.0, 7.0, 8.0, 36.0, 25.0, 34.0]

const temp_s = [298.0, 298.0, 298.0, 298.0]
const SourcePress_s = 101.0 .* ones(S)
const SinkPress_k = [101.0, 101.0, 8080.0, 14140.0, 15198.0, 15198.0]
const velocity = 20.0 .* ones(S, K)
const Mass_s = 44.0 .* ones(S)
const CR_sink_k = [-7.0, -5.0, 9.0, -20.0, -17.0, -26.0]
const EPS_POW = 1e-10

const nR = S
const nT = S * K * T
const nU = S * K
const nV = S * K * T
const nY = S * T
const nF = K
const Ntot = nR + nT + nU + nV + nY + nF

const rR = 1:nR
const rT = (last(rR) + 1):(last(rR) + nT)
const rU = (last(rT) + 1):(last(rT) + nU)
const rV = (last(rU) + 1):(last(rU) + nV)
const rY = (last(rV) + 1):(last(rV) + nY)
const rF = (last(rY) + 1):(last(rY) + nF)

const USE_OBJECTIVES = (:TotEmiss, :TAC, :InherSafeIndic)
const OBJLABELS = ["TotEmiss", "TAC", "InherSafeIndic"]
const KPART = 2

function unpack_vars(x::AbstractVector)
    Rvar = reshape(x[rR], S)
    Tvars = reshape(x[rT], S, K, T)
    Uvars = reshape(x[rU], S, K)
    Vvars = reshape(x[rV], S, K, T)
    Yv = reshape(x[rY], S, T)
    Fk = reshape(x[rF], K)
    return Rvar, Tvars, Uvars, Vvars, Yv, Fk
end

function set_core_bounds!(x)
    for s in 1:S
        set_lower_bound(x[rR[s]], L_s[s])
        set_upper_bound(x[rR[s]], M_s[s])
    end
    return nothing
end

function add_core_constraints!(m, x)
    Rvar, Tvars, Uvars, Vvars, Yv, Fk = unpack_vars(x)

    @constraint(m, [s = 1:S],
        Rvar[s] == sum(Tvars[s,k,t] for k in 1:K, t in 1:T) +
                   sum(Uvars[s,k] for k in 1:K) +
                   sum(Vvars[s,k,t] for k in 1:K, t in 1:T))

    @constraint(m, [s = 1:S, k = 1:K, t = 1:T],
        epsilon_t[t] * (Tvars[s,k,t] + Vvars[s,k,t]) * y_s[s] ==
        Tvars[s,k,t] * y_s_t[s,t])

    @constraint(m, [s = 1:S],
        Rvar[s] * y_s[s] ==
            sum(Tvars[s,k,t] * y_s_t[s,t] for k in 1:K, t in 1:T) +
            sum(Uvars[s,k] * y_s[s] for k in 1:K) +
            sum(Vvars[s,k,t] * Yv[s,t] for k in 1:K, t in 1:T))

    @constraint(m, [k = 1:K],
        Fk[k] == sum(Tvars[s,k,t] for s in 1:S, t in 1:T) +
                 sum(Uvars[s,k] for s in 1:S))

    @constraint(m, [k = 1:K],
        Fk[k] * Zmin_k[k] <=
            sum(Tvars[s,k,t] * y_s_t[s,t] for s in 1:S, t in 1:T) +
            sum(Uvars[s,k] * y_s[s] for s in 1:S))

    @constraint(m, [k = 1:K], Fk[k] <= Gmax_k[k])
    @constraint(m, Fk[1] >= 100.0)
    return Rvar, Tvars, Uvars, Vvars, Yv, Fk
end

function obj_TotEmiss(x)
    Rvar, Tvars, Uvars, _, _, Fk = unpack_vars(x)
    compression_pow = 1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                 for s in 1:S, k in 1:K, t in 1:T)
    pump_pow = 1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S, k in 1:K, t in 1:T)
    transport_pow = compression_pow + pump_pow
    source_emiss = sum((1 - 0.99) * Rvar[s] for s in 1:S)
    treat_emiss = sum(Tvars[s,k,t] * y_s_t[s,t] * gamma_t[t] for s in 1:S, k in 1:K, t in 1:T)
    transp_emiss = transport_pow * epsilon_p
    sink_emiss = sum(Fk[k] * Mu_k[k] for k in 1:K)
    return source_emiss + treat_emiss + transp_emiss + sink_emiss
end

function obj_TAC(x)
    _, Tvars, Uvars, _, _, Fk = unpack_vars(x)
    treat_cost = sum(Tvars[s,k,t] * y_s_t[s,t] * TranspCost_s_t[s,t]
                     for s in 1:S, k in 1:K, t in 1:T)

    cap_base = [sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T) / 224
                for s in 1:S, k in 1:K]
    cap_compressor = reshape([158902 * (max(b, 0.0) + EPS_POW)^0.84 * CRF for b in cap_base], S, K)
    oper_compressor = [24 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T) *
                       powerprice for s in 1:S, k in 1:K]
    compressor_cost = sum(cap_compressor) + sum(oper_compressor)

    dconst = [sqrt(velocity[s,k] * Mass_s[s] *
                   ((SinkPress_k[k] - SourcePress_s[s]) + PressDropPara_s_k[s,k]))
              for s in 1:S, k in 1:K]
    flow_sum = [sum(Tvars[s,k,t] for s in 1:S, t in 1:T) + sum(Uvars[s,k] for s in 1:S)
                for k in 1:K]
    diameter = [sqrt(max((4 / pi) * 8.314 * temp_s[s] * flow_sum[k] / dconst[s,k], 0.0) + EPS_POW)
                for s in 1:S, k in 1:K]
    pipe_cost = [(95230 * diameter[s,k] + 96904) * CRF for s in 1:S, k in 1:K]
    transp_cost = sum(DistSourSink_s_k[s,k] * pipe_cost[s,k] for s in 1:S, k in 1:K)
    sink_cost = sum(Fk[k] * CR_sink_k[k] for k in 1:K)

    return treat_cost + compressor_cost + transp_cost + sink_cost
end

function obj_InherSafeIndic(x)
    Rvar, Tvars, Uvars, Vvars, _, _ = unpack_vars(x)
    term_k = sum(I_tot_k[k] * (sum(Tvars[s,k,t] for s in 1:S, t in 1:T) +
                               sum(Uvars[s,k] for s in 1:S)) / Gmax_k[k] for k in 1:K)
    term_r = sum(I_tot_r[r] * sum(Rvar[s] for s in 1:S) / M_s_TOTAL for r in eachindex(I_tot_r))
    term_t = sum(I_tot_t[t] * sum(Tvars[s,k,t] + Vvars[s,k,t] for s in 1:S, k in 1:K) /
                 M_s_TOTAL for t in 1:T)
    return term_k + term_r + term_t
end

function obj_NegNetCapture(x)
    _, Tvars, Uvars, _, _, _ = unpack_vars(x)
    compression_pow = 1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                 for s in 1:S, k in 1:K, t in 1:T)
    pump_pow = 1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S, k in 1:K, t in 1:T)
    transport_pow = compression_pow + pump_pow
    f_co2 = [sum(Tvars[s,k,t] * y_s_t[s,t] for s in 1:S, t in 1:T) +
             sum(Uvars[s,k] * y_s[s] for s in 1:S) for k in 1:K]
    net_capture = sum(f_co2[k] * (1 - Mu_k[k]) for k in 1:K) -
                  sum(Tvars[s,k,t] * y_s_t[s,t] * gamma_t[t] for s in 1:S, k in 1:K, t in 1:T) -
                  (transport_pow / 24) * (epsilon_p / 1_000_000.0)
    return -net_capture
end

const OBJ_MAP = Dict(
    :TotEmiss => obj_TotEmiss,
    :TAC => obj_TAC,
    :InherSafeIndic => obj_InherSafeIndic,
)
const OBJFUNCS = [OBJ_MAP[k] for k in USE_OBJECTIVES]

function solve_single_objective_seed(obj::Symbol)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "max_cpu_time", SEED_TIME_LIMIT)
    set_optimizer_attribute(m, "print_level", 0)
    @variable(m, x[1:Ntot] >= 0)
    set_core_bounds!(x)
    for i in 1:Ntot
        set_start_value(x[i], 1e-3)
    end
    Rvar, Tvars, Uvars, Vvars, _, Fk = add_core_constraints!(m, x)

    @expression(m, CompressionPow,
        1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                   for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, PumpPow,
        1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                         for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, TransportPow, CompressionPow + PumpPow)

    @expression(m, F_CO2_k[k = 1:K],
        sum(Tvars[s,k,t] * y_s_t[s,t] for s in 1:S, t in 1:T) +
        sum(Uvars[s,k] * y_s[s] for s in 1:S))

    @expression(m, TreatCost,
        sum(Tvars[s,k,t] * y_s_t[s,t] * TranspCost_s_t[s,t]
            for s in 1:S, k in 1:K, t in 1:T))

    dconst = [sqrt(velocity[s,k] * Mass_s[s] *
                   ((SinkPress_k[k] - SourcePress_s[s]) + PressDropPara_s_k[s,k]))
              for s in 1:S, k in 1:K]
    @expression(m, FlowSum_k[k = 1:K],
        sum(Tvars[s,k,t] for s in 1:S, t in 1:T) + sum(Uvars[s,k] for s in 1:S))
    @NLexpression(m, D_s_k[s = 1:S, k = 1:K],
        sqrt(((4 / pi) * 8.314 * temp_s[s] * FlowSum_k[k]) / dconst[s,k] + EPS_POW))
    @NLexpression(m, PipeCost[s = 1:S, k = 1:K], (95230 * D_s_k[s,k] + 96904) * CRF)
    @NLexpression(m, TranspCost, sum(DistSourSink_s_k[s,k] * PipeCost[s,k] for s in 1:S, k in 1:K))

    @NLexpression(m, CapCompressorCost[s = 1:S, k = 1:K],
        158902 * ((sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T) / 224) +
                  EPS_POW)^0.84 * CRF)
    @expression(m, OperCompressorCost[s = 1:S, k = 1:K],
        24 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T) * powerprice)
    @NLexpression(m, CompressorCost,
        sum(CapCompressorCost[s,k] + OperCompressorCost[s,k] for s in 1:S, k in 1:K))
    @expression(m, SinkCost, sum(Fk[k] * CR_sink_k[k] for k in 1:K))
    @NLexpression(m, TAC, TreatCost + CompressorCost + TranspCost + SinkCost)

    @expression(m, SourceEmiss, sum((1 - 0.99) * Rvar[s] for s in 1:S))
    @expression(m, TreatEmiss, sum(Tvars[s,k,t] * y_s_t[s,t] * gamma_t[t]
                                   for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, TranspEmiss, TransportPow * epsilon_p)
    @expression(m, SinkEmiss, sum(Fk[k] * Mu_k[k] for k in 1:K))
    @expression(m, TotEmiss, SourceEmiss + TreatEmiss + TranspEmiss + SinkEmiss)

    @NLexpression(m, InherSafeIndic,
        sum(I_tot_k[k] * (sum(Tvars[s,k,t] for s = 1:S, t = 1:T) +
                          sum(Uvars[s,k] for s = 1:S)) / Gmax_k[k] for k = 1:K) +
        sum(I_tot_r[r] * sum(Rvar[s] for s = 1:S) / M_s_TOTAL for r = 1:length(I_tot_r)) +
        sum(I_tot_t[t] * sum(Tvars[s,k,t] + Vvars[s,k,t] for s = 1:S, k = 1:K) /
            M_s_TOTAL for t = 1:T))

    @expression(m, NetCapture,
        sum(F_CO2_k[k] * (1 - Mu_k[k]) for k in 1:K) -
        sum(Tvars[s,k,t] * y_s_t[s,t] * gamma_t[t] for s in 1:S, k in 1:K, t in 1:T) -
        (TransportPow / 24) * (epsilon_p / 1_000_000.0))
    @constraint(m, NetCapture >= CO2_gen * CapturePercent)

    if obj == :TotEmiss
        @objective(m, Min, TotEmiss)
    elseif obj == :TAC
        @NLobjective(m, Min, TAC)
    elseif obj == :InherSafeIndic
        @NLobjective(m, Min, InherSafeIndic)
    elseif obj == :NegNetCapture
        @objective(m, Min, -NetCapture)
    else
        error("Unknown objective: $obj")
    end

    optimize!(m)
    term = termination_status(m)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)) || result_count(m) == 0
        @warn "Seed solve failed or did not converge cleanly" objective=obj status=term
        return zeros(Ntot)
    end
    return value.(x)
end

function selected_point_generation(delta, xn)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "max_cpu_time", 30.0)
    @variable(m, x[1:Ntot] >= 0)
    set_core_bounds!(x)
    for i in 1:Ntot
        set_start_value(x[i], max(xn[i] + delta[i], 1e-6))
    end
    add_core_constraints!(m, x)
    @objective(m, Min, sum((x[i] - xn[i] - delta[i])^2 for i in 1:Ntot))
    optimize!(m)
    term = termination_status(m)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)) || result_count(m) == 0
        @warn "Projection failed or did not converge cleanly" status=term
        return max.(xn, 0.0)
    end
    return value.(x)
end

function jac_from_model(m)
    nlp = MathOptNLPModel(m)
    xref = ones(nlp.meta.nvar)
    return Matrix(jac(nlp, xref))
end

function build_delta_ineq_matrix()
    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:Ntot])
    _, Tvars, Uvars, _, _, Fk = unpack_vars(x)
    @constraint(m, [i = 1:Ntot], x[i] >= 0)
    @constraint(m, [k = 1:K],
        Fk[k] * Zmin_k[k] <=
            sum(Tvars[s,k,t] * y_s_t[s,t] for s in 1:S, t in 1:T) +
            sum(Uvars[s,k] * y_s[s] for s in 1:S))
    return jac_from_model(m)
end

function build_eq_matrix()
    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:Ntot])
    Rvar, Tvars, Uvars, Vvars, Yv, Fk = unpack_vars(x)
    @constraint(m, [s = 1:S],
        Rvar[s] == sum(Tvars[s,k,t] for k in 1:K, t in 1:T) +
                   sum(Uvars[s,k] for k in 1:K) +
                   sum(Vvars[s,k,t] for k in 1:K, t in 1:T))
    @constraint(m, [s = 1:S, k = 1:K, t = 1:T],
        epsilon_t[t] * (Tvars[s,k,t] + Vvars[s,k,t]) * y_s[s] ==
        Tvars[s,k,t] * y_s_t[s,t])
    @constraint(m, [s = 1:S],
        Rvar[s] * y_s[s] ==
            sum(Tvars[s,k,t] * y_s_t[s,t] for k in 1:K, t in 1:T) +
            sum(Uvars[s,k] * y_s[s] for k in 1:K) +
            sum(Vvars[s,k,t] * Yv[s,t] for k in 1:K, t in 1:T))
    @constraint(m, [k = 1:K],
        Fk[k] == sum(Tvars[s,k,t] for s in 1:S, t in 1:T) +
                 sum(Uvars[s,k] for s in 1:S))
    return jac_from_model(m)
end

function build_final_ineq_matrix()
    if FINAL_INEQ_MODE == "delta"
        return build_delta_ineq_matrix()
    end

    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:Ntot])
    if FINAL_INEQ_MODE == "bounds"
        @constraint(m, [i = 1:Ntot], x[i] >= 0)
        @constraint(m, [i = 1:Ntot], x[i] <= 1)
    elseif FINAL_INEQ_MODE == "full"
        _, Tvars, Uvars, _, _, Fk = unpack_vars(x)
        @constraint(m, [i = 1:Ntot], x[i] >= 0)
        @constraint(m, [k = 1:K], Fk[k] <= Gmax_k[k])
        @constraint(m, Fk[1] >= 100.0)
        @constraint(m, [k = 1:K],
            Fk[k] * Zmin_k[k] <=
                sum(Tvars[s,k,t] * y_s_t[s,t] for s in 1:S, t in 1:T) +
                sum(Uvars[s,k] * y_s[s] for s in 1:S))
    else
        error("Unknown CCUS_AUDIT_FINAL_INEQ=$(FINAL_INEQ_MODE). Use bounds, delta, or full.")
    end
    return jac_from_model(m)
end

function safe_normalize(v)
    nv = norm(v)
    return nv <= 1e-12 ? zero(v) : v / nv
end

function project_to_row_local(v, a)
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
            accum += sum(omega[i] * project_to_row_local(vec(normc[i, :]), vec(Ae[k, :]))
                         for i in axes(ce, 1)) / denom
        end
        for k in axes(de, 1)
            accum += sum(omega[i] * project_to_row_local(vec(normc[i, :]), vec(de[k, :]))
                         for i in axes(ce, 1)) / denom
        end
        return accum / (size(Ae, 1) + size(de, 1))
    end

    accum = zeros(size(ce, 2))
    for k in axes(Ae, 1)
        accum += sum(omega[i] * project_to_row_local(vec(normc[i, :]), vec(Ae[k, :]))
                     for i in axes(ce, 1)) / denom
    end
    return accum / size(Ae, 1)
end

function objective_matrix(objfuncs, selected_points)
    C = zeros(length(objfuncs) * length(selected_points), Ntot)
    row = 1
    for f in objfuncs
        for sp in selected_points
            C[row, :] = ForwardDiff.gradient(f, sp)
            row += 1
        end
    end
    return C
end

function generate_selected_points()
    seed_objectives = [:TotEmiss, :TAC, :InherSafeIndic, :NegNetCapture]
    vertex_list = [solve_single_objective_seed(obj) for obj in seed_objectives]
    if APPEND_ZERO_SEEDS
        for _ in 1:length(OBJFUNCS)
            push!(vertex_list, zeros(Ntot))
        end
    end

    Amats_delta = build_delta_ineq_matrix()
    Dmats = build_eq_matrix()
    selected_points = Vector{Vector{Float64}}()

    for (a, vertex) in enumerate(vertex_list)
        push!(selected_points, vertex)
        for step_id in 1:CCUS_ITERA
            sp = selected_points[end]
            Cstep = objective_matrix(OBJFUNCS, [sp])
            delta = STEP_SCALE * delta_generation(Amats_delta, Cstep, Dmats)
            newpt = selected_point_generation(delta, sp)
            push!(selected_points, newpt)
            println("CCUS seed ", a, "/", length(vertex_list), " step ", step_id, "/", CCUS_ITERA,
                    " delta_norm=", round(norm(delta), digits=4))
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
    for j in 1:Ntot
        df[!, Symbol("x$j")] = [p[j] for p in selected_points]
    end
    CSV.write(path, df)
end

function run()
    selected_points = generate_selected_points()
    Amats = build_final_ineq_matrix()
    Dmats = build_eq_matrix()
    Cmat = objective_matrix(OBJFUNCS, selected_points)

    A_current, W_current, SW_current = current_code_correlation(Amats, Cmat, Dmats, length(selected_points))
    A_star, W_star, SW_star = paper_star_correlation(Amats, Cmat, Dmats, length(selected_points))
    A_gradient = raw_gradient_cosine_correlation(Cmat, length(selected_points))

    part_current = reduced_groups(A_current, KPART)
    part_star = reduced_groups(A_star, KPART)
    part_gradient = reduced_groups(A_gradient, KPART)
    sets_current = group_sets(part_current)
    sets_star = group_sets(part_star)
    sets_gradient = group_sets(part_gradient)

    CSV.write(joinpath(OUTDIR, "audit_ccus_current_matrix.csv"), matrix_dataframe(A_current))
    CSV.write(joinpath(OUTDIR, "audit_ccus_paper_star_matrix.csv"), matrix_dataframe(A_star))
    CSV.write(joinpath(OUTDIR, "audit_ccus_gradient_cosine_matrix.csv"), matrix_dataframe(A_gradient))
    CSV.write(joinpath(OUTDIR, "audit_ccus_current_total_weight.csv"), matrix_dataframe(W_current))
    CSV.write(joinpath(OUTDIR, "audit_ccus_paper_star_total_weight.csv"), matrix_dataframe(W_star))
    CSV.write(joinpath(OUTDIR, "audit_ccus_current_strength_weight.csv"), matrix_dataframe(SW_current))
    CSV.write(joinpath(OUTDIR, "audit_ccus_paper_star_strength_weight.csv"), matrix_dataframe(SW_star))
    write_selected_points(joinpath(OUTDIR, "audit_ccus_selected_points.csv"), selected_points)

    summary = DataFrame(
        case = ["CCUS", "CCUS", "CCUS"],
        operator = ["current-code", "paper-star", "gradient-cosine"],
        objectives = [string(USE_OBJECTIVES), string(USE_OBJECTIVES), string(USE_OBJECTIVES)],
        selected_points = [length(selected_points), length(selected_points), length(selected_points)],
        iterations_per_seed = [CCUS_ITERA, CCUS_ITERA, CCUS_ITERA],
        step_scale = [STEP_SCALE, STEP_SCALE, STEP_SCALE],
        appended_zero_seeds = [APPEND_ZERO_SEEDS, APPEND_ZERO_SEEDS, APPEND_ZERO_SEEDS],
        final_ineq_mode = [FINAL_INEQ_MODE, FINAL_INEQ_MODE, FINAL_INEQ_MODE],
        requested_groups = [KPART, KPART, KPART],
        partition = [string(part_current), string(part_star), string(part_gradient)],
        group_sets = [string(sets_current), string(sets_star), string(sets_gradient)],
    )
    CSV.write(joinpath(OUTDIR, "audit_ccus_current_vs_paper_star_summary.csv"), summary)

    println(summary)
    println("current-code matrix:")
    show(stdout, "text/plain", A_current)
    println("\n\npaper-star matrix:")
    show(stdout, "text/plain", A_star)
    println("\n\ngradient-cosine matrix:")
    show(stdout, "text/plain", A_gradient)
    println("\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
