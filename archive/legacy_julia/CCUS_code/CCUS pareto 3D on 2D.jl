############# PACKAGES #############
using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX, LinearAlgebra, Ipopt, BARON, Plots,MathOptInterface

############# DATA (your definitions) #############
S = 4
M_s = [357, 1260, 399, 3426]
L_s = zeros(S)
y_s = [1, 0.44, 0.27, 0.07]
CO2_gen = sum(M_s .* y_s)
y_s_t = [1, 0.999, 0.999, 0.999]
K = 6
Zmin_k = [0.06,0.94,0.94,0.999,0.999,0.999]
Gmax_k = [126,508,3168,756,543,913]
L_s_k = zeros(S,K)
M_s_k = [126 508 3168 756 543 913;
         126 508 3168 756 543 913;
         126 508 3168 756 543 913;
         126 508 3168 756 543 913]
Mu_k = [0.28,0.5,0,0.17,0.11,0.34]
ComprePowPara_s_k = [0.43 2.22 4.37 4.37 4.37 4.37;
                     0.49 2.23 4.37 4.37 4.37 4.37;
                     0.62 2.28 4.37 4.37 4.37 4.37;
                     0.58 2.28 4.37 4.37 4.37 4.37]
PumpPowPara_s_k   = [0 0 0.204 0.017 0.176 0.204;
                     0 0 0.204 0.017 0.176 0.204;
                     0 0 0.205 0.018 0.176 0.203;
                     0 0 0.205 0.018 0.175 0.203]
PressDropPara_s_k = [56.39 832.13 49.51 49.51 50.82 51.15;
                     67.87 843.61 60.98 60.98 62.30 62.62;
                     90.82 896.07 96.72 96.72 29.84 16.72;
                     82.95 888.20 88.85 88.85 21.64 26.89]
DistSourSink_s_k  = [1.72 25.38 1.51 1.51 1.55 1.56;
                     2.07 25.73 1.86 1.86 1.9  1.91;
                     2.77 27.33 2.95 2.95 0.91 0.51;
                     2.53 27.09 2.71 2.71 0.66 0.82]

T = 1
R = 1

epsilon_t = [0.9]
epsilon_p = 0.366   # kg CO₂/kWh
powerprice = 0.02
CRF = 0.15
gamma_t = [0.0338]
CapturePercent = 0.4
TranspCost_s_t = [0,29,43,35]

I_tot_r = [12]
I_tot_t = [19]
I_tot_k = [7,7,8,36,25,34]

temp_s = [298,298,298,298]
SourcePress_s = 101*ones(S)
SinkPress_k = [101,101,8080,14140,15198,15198]

velocity = 20*ones(S,K)
Mass_s = 44*ones(S)

CR_sink_k = [-7,-5,9,-20,-17,-26]
theta_s = 0.99*ones(S)

############# MODEL BUILDER #############
"""
    build_model(; solver=:BARON)

Constructs a fresh JuMP model with all variables, constraints, and expressions.
Returns a NamedTuple: (m, TAC, TotEmiss, InherSafeIndic).
"""
function build_model(; solver=:BARON)
    m = if solver == :BARON
        model = Model(BARON.Optimizer)
        # Increase time limit and set some other helpful parameters
        set_optimizer_attribute(model, "maxtime", 30)  # 60 seconds per solve
        set_optimizer_attribute(model, "epsa", 1e-6)   # Absolute convergence tolerance
        set_optimizer_attribute(model, "epsr", 1e-6)   # Relative convergence tolerance
        set_optimizer_attribute(model, "threads", 4)   # Use parallel processing if available
        model # return the model
    else
        Model(Ipopt.Optimizer)
    end

    # Variables
    @variable(m, R_s[s=1:S], lower_bound=L_s[s], upper_bound=M_s[s])
    @variable(m, T_s_k_t[1:S,1:K,1:T] >= 0)
    @variable(m, U_s_k[1:S,1:K] >= 0)
    @variable(m, V_s_k_t[1:S,1:K,1:T] >= 0)
    @variable(m, y_s_v[1:S,1:T] >= 0)
    @variable(m, F_k[1:K] >= 0)

    # Mass balances
    @constraint(m, totMassBalance[s=1:S],
        R_s[s] == sum(T_s_k_t[s,k,t] for t in 1:T, k in 1:K) +
                  sum(U_s_k[s,k] for k in 1:K) +
                  sum(V_s_k_t[s,k,t] for t in 1:T, k in 1:K))
    @constraint(m, carbonBalan1[s=1:S,k=1:K,t=1:T],
        epsilon_t[t]*(T_s_k_t[s,k,t] + V_s_k_t[s,k,t])*y_s[s] == T_s_k_t[s,k,t]*y_s_t[s])
    @constraint(m, carbonBalan2[s=1:S,k=1:K,t=1:T],
        R_s[s]*y_s[s] ==
            sum(T_s_k_t[s,k,t]*y_s_t[s] for t in 1:T, k in 1:K) +
            sum(U_s_k[s,k]*y_s[s] for k in 1:K) +
            sum(V_s_k_t[s,k,t]*y_s_v[s,t] for t in 1:T, k in 1:K))

    # Sink constraints
    @constraint(m, sinkTotMassBalance[k=1:K],
        F_k[k] == sum(T_s_k_t[s,k,t] for t in 1:T, s in 1:S) + sum(U_s_k[s,k] for s in 1:S))
    @constraint(m, sinkMinConcenRequ[k=1:K],
        F_k[k]*Zmin_k[k] <= sum(sum(T_s_k_t[s,k,t]*y_s_t[s] for t in 1:T) for s in 1:S) +
                             sum(U_s_k[s,k]*y_s[s] for s in 1:S))
    @constraint(m, MaxSinkFlow[k=1:K], F_k[k] <= Gmax_k[k])
    @constraint(m, F_k[1] >= 100)

    # Power terms
    @expression(m, CompressionPow, 1000*sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, PumpPow, 1000*0.8*sum(PumpPowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T))
    @expression(m, TransportPow, CompressionPow + PumpPow)

    # Capture requirement
    @expression(m, F_CO2_k[k=1:K], sum(T_s_k_t[s,k,t]*y_s_t[s] for t in 1:T, s in 1:S) + sum(U_s_k[s,k]*y_s[s] for s in 1:S))
    @expression(m, NetCapture,
        sum(F_CO2_k[k]*(1-Mu_k[k]) for k in 1:K)
        - sum(sum(sum(T_s_k_t[s,k,t]*y_s_t[s]*gamma_t[t] for t in 1:T) for k in 1:K) for s in 1:S)
        - TransportPow/24 * epsilon_p / 1_000_000
    )
    @constraint(m, CarbonCapGoal, NetCapture >= CO2_gen*CapturePercent)

    # ISI
    @NLexpression(m, InherSafeIndic,
        sum(I_tot_k[k]*(sum(T_s_k_t[s,k,t] for s in 1:S,t in 1:T)+sum(U_s_k[s,k] for s in 1:S))/Gmax_k[k] for k in 1:K)
      + sum(I_tot_r[r]*sum(R_s[s] for s in 1:S)/sum(M_s[s] for s in 1:S) for r in 1:R)
      + sum(I_tot_t[t]*sum(T_s_k_t[s,k,t]+V_s_k_t[s,k,t] for s in 1:S,k in 1:K)/sum(M_s[s] for s in 1:S) for t in 1:T)
    )

    # Costs
    @expression(m, TreatCost, sum(T_s_k_t[s,k,t]*y_s_t[s]*TranspCost_s_t[s] for s in 1:S, k in 1:K, t in 1:T))
    @NLexpression(m, CapCompressorCost[s=1:S,k=1:K], 158902*((sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for t in 1:T)/224)^0.84)*CRF)
    @expression(m, OperCompressorCost[s=1:S,k=1:K], 24*sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for t in 1:T)*powerprice)
    @NLexpression(m, CompressorCost[s=1:S,k=1:K], CapCompressorCost[s,k] + OperCompressorCost[s,k])
    @expression(m, CapPumpCost[s=1:S,k=1:K], (1.11*(10^3)*sum(PumpPowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for t in 1:T) + 70000)*CRF)
    @expression(m, OperPumpCost[s=1:S,k=1:K], 24*0.8*PumpPowPara_s_k[s,k]*(sum(T_s_k_t[s,k,t]+U_s_k[s,k] for t in 1:T) + 70000)*CRF)
    @expression(m, PumpCost[s=1:S,k=1:K], CapPumpCost[s,k] + OperPumpCost[s,k])
    @NLexpression(m, compressionCost, sum(CompressorCost[s,k] + PumpCost[s,k] for s in 1:S, k in 1:K))

    # Pipe sizing helper
    PressureDiffMatrix = zeros(S,K)
    for s in 1:S, k in 1:K
        PressureDiffMatrix[s,k] = SinkPress_k[k] - SourcePress_s[s]
    end
    @NLexpression(m, SqrtTerm[s=1:S, k=1:K], sum(T_s_k_t[s,k,t] for s in 1:S, t in 1:T) + sum(U_s_k[s,k] for s in 1:S))
    DconstMatrix = zeros(S,K)
    for s in 1:S, k in 1:K
        DconstMatrix[s,k] = sqrt(velocity[s,k]*Mass_s[s]*(PressureDiffMatrix[s,k]+PressDropPara_s_k[s,k]))
    end
    @NLexpression(m, D_s_k[s=1:S, k=1:K], ((4/pi)*8.314*temp_s[s]*SqrtTerm[s,k] / DconstMatrix[s,k])^0.5 )
    @NLexpression(m, PipeCost[s=1:S,k=1:K], (95230*D_s_k[s,k] + 96904)*CRF)

    @NLexpression(m, TranspCost, sum(DistSourSink_s_k[s,k]*PipeCost[s,k] for s in 1:S, k in 1:K))
    @expression(m, SinkCost, sum(F_k[k]*CR_sink_k[k] for k in 1:K))
    @NLexpression(m, TAC, TreatCost + compressionCost + TranspCost + SinkCost)

    # Emissions
    @expression(m, SourceEmiss, sum((1-theta_s[s])*R_s[s] for s in 1:S))
    @expression(m, TreatEmiss[t=1:T], sum(T_s_k_t[s,k,t]*y_s_t[s]*gamma_t[t] for s in 1:T, k in 1:K))
    @expression(m, TranspEmiss, TransportPow*epsilon_p)
    @expression(m, SinkEmiss, sum(F_k[k]*Mu_k[k] for k in 1:K))
    @expression(m, TotEmiss, SourceEmiss + sum(TreatEmiss) + TranspEmiss + SinkEmiss)

    return (m=m, TAC=TAC, TotEmiss=TotEmiss, InherSafeIndic=InherSafeIndic)
end

############# ONE-SHOT SOLVER #############
"""
    solve_once(objective; eps_ISI=nothing, eps_Emiss=nothing, eps_Cost=nothing, solver=:BARON)

Builds a fresh model, sets ε-constraints, optimizes the chosen objective (:emiss | :isi | :cost),
returns (status, TAC, Emiss, ISI).
"""
function solve_once(objective; eps_ISI=nothing, eps_Emiss=nothing, solver=:BARON)
    # build model
    bm = build_model(solver=solver)
    m = bm.m

    # apply epsilon constraints if provided
    if eps_ISI !== nothing
        @NLconstraint(m, bm.InherSafeIndic <= eps_ISI)
    end
    if eps_Emiss !== nothing
        @NLconstraint(m, bm.TotEmiss <= eps_Emiss)
    end

    # set objective
    if objective == :cost
        @NLobjective(m, Min, bm.TAC)
    elseif objective == :isi
        @NLobjective(m, Min, bm.InherSafeIndic)
    elseif objective == :emiss
        @NLobjective(m, Min, bm.TotEmiss)
    else
        error("Unknown objective: $objective")
    end

    # optimize
    optimize!(m)

    st = termination_status(m)
    
    # Check if we have a solution before trying to read values
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
        if has_values(m)
            try
                TAC_val = objective_value(m)
                Emiss_val = value(bm.TotEmiss)
                ISI_val = value(bm.InherSafeIndic)
            catch
                TAC_val = NaN
                Emiss_val = NaN
                ISI_val = NaN
            end
        else
            TAC_val = NaN
            Emiss_val = NaN
            ISI_val = NaN
        end
    else
        TAC_val = NaN
        Emiss_val = NaN
        ISI_val = NaN
    end

    return (status=st, TAC=TAC_val, Emiss=Emiss_val, ISI=ISI_val, model=m)
end

function compute_pareto_3d(; n_ISI=20, n_Emiss=20, ISI_range=(9.0,12.0), Emiss_range=(7e5,8.75e5), solver=:BARON)
    isi_vals = range(ISI_range[1], ISI_range[2], length=n_ISI)
    emiss_vals = range(Emiss_range[1], Emiss_range[2], length=n_Emiss)

    rows = Vector{Dict{String,Any}}()

    total = length(isi_vals)*length(emiss_vals)
    i = 0
    feasible_count = 0
    
    for eps_isi in isi_vals
        for eps_em in emiss_vals
            i += 1
            println("[$i/$total] Solving for eps_ISI=$(round(eps_isi,digits=4)), eps_Emiss=$(round(eps_em,digits=2))...")
            
            res = solve_once(:cost, eps_ISI=eps_isi, eps_Emiss=eps_em, solver=solver)
            
            if res.status == MOI.OPTIMAL || res.status == MOI.LOCALLY_SOLVED
                feasible_count += 1
            end
            
            push!(rows, Dict(
                "eps_ISI" => eps_isi,
                "eps_Emiss" => eps_em,
                "status" => string(res.status),
                "TAC" => res.TAC,
                "Emiss" => res.Emiss,
                "ISI" => res.ISI
            ))
            
            if i % 10 == 0
                println("Progress: $i/$total points computed ($feasible_count feasible)")
            end
        end
    end

    df = DataFrame(rows)
    println("\nCompleted: $feasible_count/$total points were feasible")
    return df
end

function save_results(df::DataFrame, filepath::AbstractString="pareto_3d_results.csv")
    # Filter to just the feasible solutions before saving
    df_feasible = filter(row -> !isnan(row.TAC), df)
    CSV.write(filepath, df_feasible)
    println("Saved $(nrow(df_feasible)) feasible solutions to: $filepath")
end

# small runnable CLI when script invoked directly

# default grid using the approximate ranges you provided
df = compute_pareto_3d(n_ISI=4, n_Emiss=4, 
                        ISI_range=(9.0,12.0), 
                        Emiss_range=(7e5,8.75e5), 
                        solver=:BARON)
save_results(df, "pareto_3d_results.csv")
println("\nFirst 10 feasible solutions:")
println(filter(row -> !isnan(row.TAC), df)[1:min(10,end), :])
