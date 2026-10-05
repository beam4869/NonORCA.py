using JuMP, BARON
# using Ipopt  # optional alternative solver

# ---------------------------
# Data (as given)
# ---------------------------
S = 4  # sources
M_s = [357.0, 1260.0, 399.0, 3426.0]
L_s = zeros(S)
y_s = [1.0, 0.44, 0.27, 0.07]
CO2_gen = sum(M_s .* y_s)
y_s_t = [1.0, 0.999, 0.999, 0.999]  # purity after treatment (T=1)

K = 6  # sinks
Zmin_k = [0.06, 0.94, 0.94, 0.999, 0.999, 0.999]
Gmax_k = [126.0, 508.0, 3168.0, 756.0, 543.0, 913.0]
L_s_k = zeros(S, K)
M_s_k = [126 508 3168 756 543 913;
         126 508 3168 756 543 913;
         126 508 3168 756 543 913;
         126 508 3168 756 543 913] .|> float

Mu_k = [0.28, 0.5, 0.0, 0.17, 0.11, 0.34]  # sink inefficiency

ComprePowPara_s_k = [0.43 2.22 4.37 4.37 4.37 4.37;
                     0.49 2.23 4.37 4.37 4.37 4.37;
                     0.62 2.28 4.37 4.37 4.37 4.37;
                     0.58 2.28 4.37 4.37 4.37 4.37] .|> float

PumpPowPara_s_k = [0 0 0.204 0.017 0.176 0.204;
                   0 0 0.204 0.017 0.176 0.204;
                   0 0 0.205 0.018 0.176 0.203;
                   0 0 0.205 0.018 0.175 0.203] .|> float

PressDropPara_s_k = [56.39 832.13 49.51 49.51 50.82 51.15;
                     67.87 843.61 60.98 60.98 62.30 62.62;
                     90.82 896.07 96.72 96.72 29.84 16.72;
                     82.95 888.20 88.85 88.85 21.64 26.89] .|> float

DistSourSink_s_k = [1.72 25.38 1.51 1.51 1.55 1.56;
                    2.07 25.73 1.86 1.86 1.90 1.91;
                    2.77 27.33 2.95 2.95 0.91 0.51;
                    2.53 27.09 2.71 2.71 0.66 0.82] .|> float

T = 1                 # number of treatment technologies
Ropt = 1              # number of transport options (renamed from R to avoid collision)
epsilon_t = [0.9]
epsilon_p = 0.366     # kg CO2/kWh
powerprice = 0.02     # USD/kWh
CRF = 0.15
gamma_t = [0.0338]
CapturePercent = 0.4
TranspCost_s_t = [0.0, 29.0, 43.0, 35.0]  # cost indexed by source s (T=1)

# Safety indices
I_tot_r = [12.0]
I_tot_t = [19.0]
I_tot_k = [7.0, 7.0, 8.0, 36.0, 25.0, 34.0]

# Pressure/temperature data for pipe sizing surrogate
temp_s = [298.0, 298.0, 298.0, 298.0]
SourcePress_s = 101.0 .* ones(S)
SinkPress_k = [101.0, 101.0, 8080.0, 14140.0, 15198.0, 15198.0]
PressureDiffMatrix = zeros(S, K)
for s in 1:S, k in 1:K
    PressureDiffMatrix[s, k] = SinkPress_k[k] - SourcePress_s[s]
end
velocity = 20.0 .* ones(S, K)
Mass_s = 44.0 .* ones(S)  # CO2 MW as placeholder

# ---------------------------
# Model with one decision vector x
# ---------------------------
CCUS = Model(BARON.Optimizer)
set_optimizer_attribute(CCUS, "maxtime", 60)
# set_optimizer(CCUS, Ipopt.Optimizer)

# ---- sizes (Ints only)
nR   = S           # R_s
nT   = S*K*T       # T_s_k_t
nU   = S*K         # U_s_k
nV   = S*K*T       # V_s_k_t
nYv  = S*T         # y_s_v
nF   = K           # F_k
Ntot = nR + nT + nU + nV + nYv + nF

# ---- contiguous integer ranges (compute BEFORE any JuMP vars)
rR  = 1:nR
rT  = (last(rR)+1):(last(rR)+nT)
rU  = (last(rT)+1):(last(rT)+nU)
rV  = (last(rU)+1):(last(rU)+nV)
rYv = (last(rV)+1):(last(rV)+nYv)
rF  = (last(rYv)+1):(last(rYv)+nF)

# ---- single decision vector
@variable(CCUS, x[1:Ntot] >= 0)

# ---- views (reshaped slices of x)
R  = reshape(x[rR],  S)            # R_s[s]
T_ = reshape(x[rT],  S, K, T)      # T_s_k_t[s,k,t]
U_ = reshape(x[rU],  S, K)         # U_s_k[s,k]
V_ = reshape(x[rV],  S, K, T)      # V_s_k_t[s,k,t]
Yv = reshape(x[rYv], S, T)         # y_s_v[s,t]
F  = reshape(x[rF],  K)            # F_k[k]

# Bounds for R
for s in 1:S
    set_lower_bound(R[s], L_s[s])
    set_upper_bound(R[s], M_s[s])
end

# ---------------------------
# Constraints
# ---------------------------

# 1) Total mass balance at sources
@constraint(CCUS, [s=1:S],
    R[s] == sum(T_[s,k,t] for k in 1:K, t in 1:T) +
            sum(U_[s,k]   for k in 1:K) +
            sum(V_[s,k,t] for k in 1:K, t in 1:T)
)

# 2) Carbon balances
@constraint(CCUS, [s=1:S, k=1:K, t=1:T],
    epsilon_t[t] * (T_[s,k,t] + V_[s,k,t]) * y_s[s] == T_[s,k,t] * y_s_t[s]
)

@constraint(CCUS, [s=1:S],
    R[s]*y_s[s] ==
        sum(T_[s,k,t]*y_s_t[s] for k in 1:K, t in 1:T) +
        sum(U_[s,k]*y_s[s]     for k in 1:K) +
        sum(V_[s,k,t]*Yv[s,t]  for k in 1:K, t in 1:T)
)

# 3) Sink total flow and concentration requirement
@constraint(CCUS, [k=1:K],
    F[k] == sum(T_[s,k,t] for s in 1:S, t in 1:T) + sum(U_[s,k] for s in 1:S)
)

@constraint(CCUS, [k=1:K],
    F[k]*Zmin_k[k] <=
        sum(T_[s,k,t]*y_s_t[s] for s in 1:S, t in 1:T) +
        sum(U_[s,k]*y_s[s]     for s in 1:S)
)

@constraint(CCUS, [k=1:K], F[k] <= Gmax_k[k])
@constraint(CCUS, F[1] >= 100.0)

# ---------------------------
# Expressions & Nonlinear parts
# ---------------------------

# Transport power
@expression(CCUS,
    CompressionPow,
    1000 * sum(ComprePowPara_s_k[s,k] * (T_[s,k,t] + U_[s,k]) for s in 1:S, k in 1:K, t in 1:T)
)
@expression(CCUS,
    PumpPow,
    1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (T_[s,k,t] + U_[s,k]) for s in 1:S, k in 1:K, t in 1:T)
)
@expression(CCUS, TransportPow, CompressionPow + PumpPow)

# CO2 into each sink
@expression(CCUS, F_CO2_k[k=1:K],
    sum(T_[s,k,t]*y_s_t[s] for s in 1:S, t in 1:T) +
    sum(U_[s,k]*y_s[s]     for s in 1:S)
)

# Net capture (constraint)
@expression(CCUS,
    NetCapture,
    sum(F_CO2_k[k]*(1 - Mu_k[k]) for k in 1:K)
    - sum(sum(T_[s,k,t]*y_s_t[s]*gamma_t[t] for t in 1:T, k in 1:K) for s in 1:S)
    - TransportPow/24 * epsilon_p / 1_000_000
)
@constraint(CCUS, NetCapture >= CO2_gen * CapturePercent)

# Inherent safety indicator (with Ropt)
total_Ms = sum(M_s) 
@NLexpression(CCUS,
    InherSafeIndic,
    sum(I_tot_k[k] * (sum(T_[s,k,t] for s in 1:S, t in 1:T) + sum(U_[s,k] for s in 1:S)) / Gmax_k[k] for k in 1:K)
    + sum(I_tot_r[r] * sum(R[s] for s in 1:S) / total_Ms for r in 1:Ropt)
    + sum(I_tot_t[t] * sum(T_[s,k,t] + V_[s,k,t] for s in 1:S, k in 1:K) / total_Ms for t in 1:T)
)

# Treatment & compression/pump costs
@expression(CCUS,
    TreatCost,
    sum(T_[s,k,t]*y_s_t[s]*TranspCost_s_t[s] for s in 1:S, k in 1:K, t in 1:T)
)

@NLexpression(CCUS,
    CapCompressorCost[s=1:S, k=1:K],
    158902 * ((sum(ComprePowPara_s_k[s,k]*(T_[s,k,t] + U_[s,k]) for t in 1:T)/224)^0.84) * CRF
)
@expression(CCUS,
    OperCompressorCost[s=1:S, k=1:K],
    24 * sum(ComprePowPara_s_k[s,k]*(T_[s,k,t] + U_[s,k]) for t in 1:T) * powerprice
)
@NLexpression(CCUS, CompressorCost[s=1:S,k=1:K], CapCompressorCost[s,k] + OperCompressorCost[s,k])

@expression(CCUS,
    CapPumpCost[s=1:S, k=1:K],
    (1.11*1e3 * sum(PumpPowPara_s_k[s,k]*(T_[s,k,t] + U_[s,k]) for t in 1:T) + 70000) * CRF
)
@expression(CCUS,
    OperPumpCost[s=1:S, k=1:K],
    24*0.8*PumpPowPara_s_k[s,k] * (sum(T_[s,k,t] + U_[s,k] for t in 1:T) + 70000) * CRF
)
@expression(CCUS, PumpCost[s=1:S, k=1:K], CapPumpCost[s,k] + OperPumpCost[s,k])

@NLexpression(CCUS, compressionCost, sum(CompressorCost[s,k] + PumpCost[s,k] for s=1:S, k=1:K))

# Pipe sizing surrogate & cost
DconstMatrix = zeros(S, K)
for s in 1:S, k in 1:K
    DconstMatrix[s, k] = sqrt(velocity[s, k] * Mass_s[s] * (PressureDiffMatrix[s, k] + PressDropPara_s_k[s, k]))
end

@NLexpression(CCUS, SqrtTerm[s=1:S, k=1:K],
    sum(T_[s,k,t] for t in 1:T, s in 1:S) + sum(U_[s,k] for s in 1:S)   # <- fixed: no accidental sum over s
)

@NLexpression(CCUS, D_s_k[s=1:S, k=1:K],
    ((4/pi) * 8.314 * temp_s[s] * SqrtTerm[s,k] / DconstMatrix[s,k])^0.5
)

@NLexpression(CCUS, PipeCost[s=1:S, k=1:K],
    (95230 * D_s_k[s,k] + 96904) * CRF
)

@NLexpression(CCUS, TranspCost,
    sum(DistSourSink_s_k[s,k] * PipeCost[s,k] for s in 1:S, k in 1:K)
)

CR_sink_k = [-7.0, -5.0, 9.0, -20.0, -17.0, -26.0]
@expression(CCUS, SinkCost, sum(F[k] * CR_sink_k[k] for k in 1:K))

@NLexpression(CCUS, TAC, TreatCost + compressionCost + TranspCost + SinkCost)

# Emissions (optional diagnostics)
theta_s = 0.99 .* ones(S)
@expression(CCUS, SourceEmiss, sum((1 - theta_s[s]) * R[s] for s in 1:S))
@expression(CCUS, TreatEmiss[t=1:T], sum(T_[s,k,t] * y_s_t[s] * gamma_t[t] for s in 1:S, k in 1:K))
@expression(CCUS, TranspEmiss, TransportPow * epsilon_p)
@expression(CCUS, SinkEmiss, sum(F[k] * Mu_k[k] for k in 1:K))
@expression(CCUS, TotEmiss, SourceEmiss + sum(TreatEmiss) + TranspEmiss + SinkEmiss)

# ---------------------------
# Objective and solve
# ---------------------------
@NLobjective(CCUS, Min, TAC)
# @NLobjective(CCUS, Min,TotEmiss)
# @NLobjective(CCUS, Min,InherSafeIndic)
optimize!(CCUS)

println("Termination status: ", termination_status(CCUS))
if has_values(CCUS)
    println("Objective (TAC): ", value(TAC))
    println("Net capture: ", value(NetCapture))
    println("Total emissions: ", value(TotEmiss))
    println("Inherent safety indicator: ", value(InherSafeIndic))
end

# ---------------------------
# Retrieve and unpack x
# ---------------------------
xstar = has_values(CCUS) ? value.(x) : nothing
if xstar !== nothing
    println("x length = ", length(xstar))
    Rstar  = reshape(xstar[rR],  S)
    Tstar  = reshape(xstar[rT],  S, K, T)
    Ustar  = reshape(xstar[rU],  S, K)
    Vstar  = reshape(xstar[rV],  S, K, T)
    Yvstar = reshape(xstar[rYv], S, T)
    Fstar  = reshape(xstar[rF],  K)

    println("R*: ", Rstar)
    println("F*: ", Fstar)
end
