############################
# Single-objective seeds
############################
using JuMP, Ipopt, BARON
import MathOptInterface as MOI

# ---------- Data ----------
S = 4
M_s = [357.0, 1260.0, 399.0, 3426.0]
L_s = zeros(S)
y_s = [1.0, 0.44, 0.27, 0.07]
CO2_gen = sum(M_s .* y_s)

y_s_t = [1.0, 0.999, 0.999, 0.999] # per-source purity after treatment (T=1)
K = 6
Zmin_k = [0.06, 0.94, 0.94, 0.999, 0.999, 0.999]
Gmax_k = [126.0, 508.0, 3168.0, 756.0, 543.0, 913.0]
L_s_k = zeros(S, K)
M_s_k = [126 508 3168 756 543 913;
         126 508 3168 756 543 913;
         126 508 3168 756 543 913;
         126 508 3168 756 543 913] .|> float
Mu_k = [0.28, 0.5, 0.0, 0.17, 0.11, 0.34]

ComprePowPara_s_k = [0.43 2.22 4.37 4.37 4.37 4.37;
                     0.49 2.23 4.37 4.37 4.37 4.37;
                     0.62 2.28 4.37 4.37 4.37 4.37;
                     0.58 2.28 4.37 4.37 4.37 4.37] .|> float
PumpPowPara_s_k   = [0 0 0.204 0.017 0.176 0.204;
                     0 0 0.204 0.017 0.176 0.204;
                     0 0 0.205 0.018 0.176 0.203;
                     0 0 0.205 0.018 0.175 0.203] .|> float
PressDropPara_s_k = [56.39 832.13 49.51 49.51 50.82 51.15;
                     67.87 843.61 60.98 60.98 62.30 62.62;
                     90.82 896.07 96.72 96.72 29.84 16.72;
                     82.95 888.20 88.85 88.85 21.64 26.89] .|> float
DistSourSink_s_k  = [1.72 25.38 1.51 1.51 1.55 1.56;
                     2.07 25.73 1.86 1.86 1.90 1.91;
                     2.77 27.33 2.95 2.95 0.91 0.51;
                     2.53 27.09 2.71 2.71 0.66 0.82] .|> float

T = 1              # number of treatment technologies
R = 1              # number of transport options
epsilon_t = [0.9]
epsilon_p = 0.366  # kg CO2 / kWh
powerprice = 0.02  # USD / kWh
CRF = 0.15
gamma_t = [0.0338]
CapturePercent = 0.4
TranspCost_s_t = [0.0, 29.0, 43.0, 35.0]
CR_sink_k = [-7.0, -5.0, 9.0, -20.0, -17.0, -26.0]

# Safety indices
I_tot_r = [12.0]
I_tot_t = [19.0]
I_tot_k = [7.0, 7.0, 8.0, 36.0, 25.0, 34.0]

# Pipe surrogate parameters
temp_s        = fill(298.0, S)
SourcePress_s = 101.0 .* ones(S)
SinkPress_k   = [101.0, 101.0, 8080.0, 14140.0, 15198.0, 15198.0]
velocity      = 20.0 .* ones(S, K)
Mass_s        = 44.0 .* ones(S)

# ---------- Helpers ----------
const EPS_POW = 1e-8
y_s_t_mat = reshape(y_s_t, S, T)
TranspCost_s_t_mat = reshape(TranspCost_s_t, S, T)

# Decision vector layout (R, T, U, V, yv, F)
nR   = S
nT   = S * K * T
nU   = S * K
nV   = S * K * T
nYv  = S * T
nF   = K
Ntot = nR + nT + nU + nV + nYv + nF

rR  = 1:nR
rT  = (last(rR)  + 1):(last(rR)  + nT)
rU  = (last(rT)  + 1):(last(rT)  + nU)
rV  = (last(rU)  + 1):(last(rU)  + nV)
rYv = (last(rV)  + 1):(last(rV)  + nYv)
rF  = (last(rYv) + 1):(last(rYv) + nF)

# ---------- Single-objective seed solver ----------
function solve_single_objective_seed(obj::Symbol; optimizer = Ipopt.Optimizer, time_limit::Real = 120.0)
    m = Model(optimizer)
    if optimizer == Ipopt.Optimizer
        set_optimizer_attribute(m, "max_cpu_time", time_limit)
        set_optimizer_attribute(m, "print_level", 0)
    elseif optimizer == BARON.Optimizer
        set_optimizer_attribute(m, "maxtime", time_limit)
        set_optimizer_attribute(m, "prlevel", 0)
    end

    @variable(m, x[1:Ntot] >= 0)
    for i in 1:Ntot
        set_start_value(x[i], 1e-3)
    end

    # Views (renamed to avoid colliding with scalar R, T)
    Rvar  = reshape(x[rR],  S)
    Tvars = reshape(x[rT],  S, K, T)
    Uvars = reshape(x[rU],  S, K)
    Vvars = reshape(x[rV],  S, K, T)
    Yv    = reshape(x[rYv], S, T)
    Fvar  = reshape(x[rF],  K)

    # ----- Constraints -----
    @constraint(m, [s=1:S],
        Rvar[s] == sum(Tvars[s,k,t] for k in 1:K, t in 1:T) +
                   sum(Uvars[s,k]   for k in 1:K) +
                   sum(Vvars[s,k,t] for k in 1:K, t in 1:T))

    @constraint(m, [s=1:S, k=1:K, t=1:T],
        epsilon_t[t] * (Tvars[s,k,t] + Vvars[s,k,t]) * y_s[s] == Tvars[s,k,t] * y_s_t_mat[s,t])

    @constraint(m, [s=1:S],
        Rvar[s]*y_s[s] ==
            sum(Tvars[s,k,t]*y_s_t_mat[s,t] for k in 1:K, t in 1:T) +
            sum(Uvars[s,k]*y_s[s]           for k in 1:K) +
            sum(Vvars[s,k,t]*Yv[s,t]        for k in 1:K, t in 1:T))

    @constraint(m, [k=1:K],
        Fvar[k] == sum(Tvars[s,k,t] for s in 1:S, t in 1:T) + sum(Uvars[s,k] for s in 1:S))

    @constraint(m, [k=1:K],
        Fvar[k] * Zmin_k[k] <=
            sum(Tvars[s,k,t]*y_s_t_mat[s,t] for s in 1:S, t in 1:T) +
            sum(Uvars[s,k]*y_s[s]           for s in 1:S))

    @constraint(m, [k=1:K], Fvar[k] <= Gmax_k[k])
    @constraint(m, Fvar[1] >= 100.0)

    # ----- Shared expressions -----
    @expression(m, CompressionPow,
        1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, PumpPow,
        1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, TransportPow, CompressionPow + PumpPow)

    @expression(m, F_CO2_k[k=1:K],
        sum(Tvars[s,k,t]*y_s_t_mat[s,t] for s in 1:S, t in 1:T) +
        sum(Uvars[s,k]*y_s[s]           for s in 1:S))

    # Treat cost (linear)
    @expression(m, TreatCost,
        sum(Tvars[s,k,t]*y_s_t_mat[s,t]*TranspCost_s_t_mat[s,t] for s in 1:S, k in 1:K, t in 1:T))

    # Pipe surrogate
    DconstMatrix = [sqrt(velocity[s,k] * Mass_s[s] *
                         ((SinkPress_k[k] - SourcePress_s[s]) + PressDropPara_s_k[s,k]))
                    for s in 1:S, k in 1:K]

    @expression(m, FlowSum_k[k=1:K],
        sum(Tvars[s1,k,t1] for s1 in 1:S, t1 in 1:T) + sum(Uvars[s1,k] for s1 in 1:S))

    @NLexpression(m, D_s_k[s=1:S, k=1:K],
        sqrt( ((4/pi) * 8.314 * temp_s[s] * FlowSum_k[k]) / DconstMatrix[s,k] + EPS_POW ))

    @NLexpression(m, PipeCost[s=1:S, k=1:K], (95230 * D_s_k[s,k] + 96904) * CRF)
    @NLexpression(m, TranspCost, sum(DistSourSink_s_k[s,k] * PipeCost[s,k] for s in 1:S, k in 1:K))

    # Compressor costs
    @NLexpression(m, CapCompressorCost[s=1:S, k=1:K],
        158902 * ((sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T) / 224) + EPS_POW)^0.84 * CRF)
    @expression(m, OperCompressorCost[s=1:S, k=1:K],
        24 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T) * powerprice)
    @NLexpression(m, CompressorCost,
        sum(CapCompressorCost[s,k] + OperCompressorCost[s,k] for s in 1:S, k in 1:K))

    # Sink cost
    @expression(m, SinkCost, sum(Fvar[k] * CR_sink_k[k] for k in 1:K))

    # TAC (nonlinear)
    @NLexpression(m, TAC, TreatCost + CompressorCost + TranspCost + SinkCost)

    # Emissions pieces & TotEmiss
    @expression(m, SourceEmiss, sum((1 - 0.99) * Rvar[s] for s in 1:S))
    @expression(m, TreatEmiss,  sum(Tvars[s,k,t]*y_s_t_mat[s,t]*gamma_t[t] for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, TranspEmiss, TransportPow * epsilon_p)
    @expression(m, SinkEmiss,   sum(Fvar[k] * Mu_k[k] for k in 1:K))
    @expression(m, TotEmiss, SourceEmiss + TreatEmiss + TranspEmiss + SinkEmiss)

    # Inherent safety (nonlinear)
    R_cnt = R
    @NLexpression(m, InherSafeIndic,
        sum(I_tot_k[k] * (sum(Tvars[s,k,t] for s=1:S, t=1:T) + sum(Uvars[s,k] for s=1:S)) / Gmax_k[k] for k=1:K) +
        sum(I_tot_r[r] * sum(Rvar[s] for s=1:S) / sum(M_s[s] for s=1:S) for r=1:R_cnt) +
        sum(I_tot_t[t] * sum(Tvars[s,k,t] + Vvars[s,k,t] for s=1:S, k=1:K) / sum(M_s[s] for s=1:S) for t=1:T)
    )

    # Net capture (linear expression here)
    @expression(m, NetCapture,
        sum(F_CO2_k[k] * (1 - Mu_k[k]) for k in 1:K) -
        sum(sum(Tvars[s,k,t]*y_s_t_mat[s,t]*gamma_t[t] for t in 1:T, k in 1:K) for s in 1:S) -
        (TransportPow / 24) * (epsilon_p / 1_000_000.0)
    )

    # Capture requirement (active during seeding)
    @constraint(m, NetCapture >= CO2_gen * CapturePercent)

    # ----- Choose objective -----
    if obj == :TotEmiss
        @objective(m, Min, TotEmiss)
    elseif obj == :TAC
        @NLobjective(m, Min, TAC)
    elseif obj == :InherSafeIndic
        @NLobjective(m, Min, InherSafeIndic)
    elseif obj == :NegNetCapture
        @objective(m, Min, -NetCapture)  # maximize capture
    else
        error("Unknown objective: $obj")
    end

    optimize!(m)
    term = termination_status(m)
    if term ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        @warn "Single-objective solve for $obj finished with status = $term"
    end
    return value.(x)
end

# ---------- Build seeds ----------
objectives_for_seeds = [:TotEmiss, :TAC, :InherSafeIndic, :NegNetCapture]
vertex_list = [solve_single_objective_seed(obj; optimizer = Ipopt.Optimizer, time_limit = 120.0)
               for obj in objectives_for_seeds]

println("Built ", length(vertex_list), " single-objective seeds.")
