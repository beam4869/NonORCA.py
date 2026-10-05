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


function build_model(; solver=:BARON)
    CCUS = if solver == :BARON
        model = Model(BARON.Optimizer)
        # Increase time limit and set some other helpful parameters
        set_optimizer_attribute(model, "maxtime", 60)  # 60 seconds per solve
        set_optimizer_attribute(model, "epsa", 1e-6)   # Absolute convergence tolerance
        set_optimizer_attribute(model, "epsr", 1e-6)   # Relative convergence tolerance
        set_optimizer_attribute(model, "threads", 8)   # Use parallel processing if available
        model # return the model
    else
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "tol", 1e-6)
        set_optimizer_attribute(model, "acceptable_tol", 1e-5)
        set_optimizer_attribute(model, "acceptable_iter", 10)
        model
    end
    eps_nl = 1e-8
    @variable(CCUS,R_s[s=1:S], lower_bound=L_s[s], upper_bound=M_s[s])
    @variable(CCUS,T_s_k_t[1:S,1:K,1:T] >=0) # treated carbon flow unit with kt/y
    @variable(CCUS,U_s_k[1:S,1:K] >=0) # carbon flow without treatment
    @variable(CCUS,V_s_k_t[1:S,1:K,1:T] >=0) # new mass balance, waste of the treated carbon flow
    @variable(CCUS,y_s_v[1:S,1:T] >=0)
    @variable(CCUS,F_k[1:K] >=0) # total flow into any sink
    #@variable(CCUS,X_s_k[1:S,1:K],Bin) # total flow into any sink
    # @variable(CCUS,P_k[1:K]>=0) # power usage in sinks

    ##########################Mass balance
    @constraint(CCUS,totMassBalance[s in 1:S], R_s[s] == sum(T_s_k_t[s,k,t] for t in 1:T, k in 1:K) + sum(U_s_k[s,k] for k in 1:K) + sum(V_s_k_t[s,k,t]  for t in 1:T, k in 1:K) ) # total mass balance
    # @constraint(CCUS,carbonMassBalance[s in 1:S], R_s[s]*y_s[s] == sum(sum(epsilon_t[t]*T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T) for k in 1:K) + sum(U_s_k[s,k]*y_s[s] for k in 1:K) ) # carbon mass balance
    @constraint(CCUS,carbonBalan1[s in 1:S,k in 1:K,t in 1:T], epsilon_t[t]*(T_s_k_t[s,k,t]+V_s_k_t[s,k,t])*y_s[s] == T_s_k_t[s,k,t]*y_s_t[s,t] ) # carbon mass balance
    @constraint(CCUS,carbonBalan2[s in 1:S,k in 1:K,t in 1:T], R_s[s]*y_s[s] == sum(T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T, k in 1:K) + sum(U_s_k[s,k]*y_s[s] for k in 1:K) +sum(V_s_k_t[s,k,t]*y_s_v[s,t] for t in 1:T, k in 1:K) ) # carbon mass balance
    #@constraint(CCUS,carbonBalan3[s in 1:S,k in 1:K,t in 1:T], (1-epsilon_t[t])*(T_s_k_t[s,k,t]+V_s_k_t[s,k,t])*y_s_t[s,t] == V_s_k_t[s,k,t]*y_s_v[s,t] ) # carbon mass balance
    ##total and carbon mass balance for sinks:
    @constraint(CCUS,sinkTotMassBalance[k in 1:K],F_k[k] == sum(T_s_k_t[s,k,t] for t in 1:T, s in 1:S)+sum(U_s_k[s,k] for s in 1:S)) 
    @constraint(CCUS,sinkMinConcenRequ[k in 1:K],F_k[k]*Zmin_k[k] <= sum(sum(T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T) for s in 1:S)+sum(U_s_k[s,k]*y_s[s] for s in 1:S)) 
    @constraint(CCUS,MaxSinkFlow[k in 1:K], F_k[k]<=Gmax_k[k])
    #@constraint(CCUS,T_s_k_t[2,3,1]>=0.1)
    @constraint(CCUS, F_k[1]>=100)
    #@constraint(CCUS,MinTandUFlow[s in 1:S,k in 1:K,t in 1:T], L_s_k[s,k]*X_s_k[s,k]<=T_s_k_t[s,k,t]+U_s_k[s,k]) #min for the total flow from s to k
    #@constraint(CCUS,MaxTandUFlow[s in 1:S,k in 1:K,t in 1:T], Mq_s_k[s,k]*X_s_k[s,k]>=T_s_k_t[s,k,t]+U_s_k[s,k]) #max for the total flow from s to k
    ###########
    @expression(CCUS, CompressionPow,1000*sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T) ) # compressor power.*1000 resulting in kwd/y
    @expression(CCUS, PumpPow,1000*0.8*sum(PumpPowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T) ) # sum of the pump power and compressor power. per year
    @expression(CCUS, TransportPow, CompressionPow + PumpPow ) # sum of the pump power and compressor power. per year
    ##########################Carbon capture requirements
    @expression(CCUS, F_CO2_k[k in 1:K], sum(T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T, s in 1:S) + sum(U_s_k[s,k]*y_s[s] for s in 1:S) ) # total flow into any sink
    @expression(CCUS, NetCapture, sum(F_CO2_k[k]*(1-Mu_k[k]) for k in 1:K) - sum( sum(sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for t in 1:T) for k in 1:K) for s in 1:S) - TransportPow/24*epsilon_p/1000000 ) #TransportPow in kwd, /24 to kwh; eplison is kg CO2/kwh, 1000000 to kt
    @constraint(CCUS,CarbonCapGoal, NetCapture >= CO2_gen*CapturePercent)#14.905*365*CapturePercent) # want the total carbon capture satisfy the set goal from Section 3 A daily capacity of 14,905 t of CO2 is available for capture, utilization, storage, or controlled emission.
    #use kt here for carbon emission
    ############################
    ##################ISI objective
    #@NLexpression(CCUS,ISISqrtExpre1,sum(T_s_k_t[s,k,t]^2 for s in 1:S, k in 1:K, t in 1:T)^0.5 )
    #@NLexpression(CCUS,ISISqrtExpre2,sum(U_s_k[s,k]^2 for s in 1:S, k in 1:K)^0.5 )
    #@NLexpression(CCUS, InherSafeIndic, sum( sum((I_tot_t[t]+I_tot_r[r]+I_tot_k[k])*T_s_k_t[s,k,t]/ISISqrtExpre1 for t in 1:T, r in 1:R) + sum((I_tot_r[r]+I_tot_k[k])*U_s_k[s,k]/ISISqrtExpre2 for r in 1:R) for s in 1:S, k in 1:K) ) #without normalize
    @expression(CCUS, InherSafeIndic, sum(I_tot_k[k]*(sum(T_s_k_t[s,k,t] for s in 1:S,t in 1:T)+sum(U_s_k[s,k] for s in 1:S))/Gmax_k[k] for k in 1:K)+sum(I_tot_r[r]*sum(R_s[s] for s in 1:S)/sum(M_s[s] for s in 1:S) for r in 1:R) + sum(I_tot_t[t]*sum(T_s_k_t[s,k,t]+V_s_k_t[s,k,t] for s in 1:S,k in 1:K)/sum(M_s[s] for s in 1:S) for t in 1:T))

    #costs:
    ############################
    @expression(CCUS, TreatCost, sum(T_s_k_t[s,k,t]*y_s_t[s,t]*TranspCost_s_t[s,t] for s in 1:S, k in 1:K, t in 1:T))
    @NLexpression(CCUS, CapCompressorCost[s in 1:S,k in 1:K], 158902*((sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for t in 1:T)/224)^0.84)*CRF) # times 365 or not
    @expression(CCUS, OperCompressorCost[s in 1:S,k in 1:K],24*sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for t in 1:T)*powerprice) ##
    @NLexpression(CCUS, CompressorCost[s in 1:S,k in 1:K], CapCompressorCost[s,k] + OperCompressorCost[s,k])
    @expression(CCUS, CapPumpCost[s in 1:S,k in 1:K], (1.11*(10^3)*sum(PumpPowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for t in 1:T) + 70000)*CRF)
    @expression(CCUS, OperPumpCost[s in 1:S,k in 1:K], 24*0.8*PumpPowPara_s_k[s,k]*(sum(T_s_k_t[s,k,t]+U_s_k[s,k] for t in 1:T) + 70000)*CRF)
    @expression(CCUS, PumpCost[s in 1:S,k in 1:K], CapPumpCost[s,k] +OperPumpCost[s,k])
    @NLexpression(CCUS,compressionCost, sum(CompressorCost[s,k] + PumpCost[s,k] for s in 1:S, k in 1:K))
    PressureDiffMatrix = zeros(S,K)
    for s in 1:S
        for k in 1:K
            PressureDiffMatrix[s,k] = SinkPress_k[k] - SourcePress_s[s]
        end
    end 
    println("pressure:",PressureDiffMatrix)
    # @NLexpression(CCUS,D_s_k[s in 1:S,k in 1:K],sqrt(((4/pi)*8.314*temp_s[s]*(sum(T_s_k_t[s,k,t] for s in 1:S, t in 1:T) + sum(U_s_k[s,k] for s in 1:S))/(velocity[s,k]*Mass_s[s]*(PressureDiffMatrix[s,k] + PressDropPara_s_k[s,k])) )) # D_s_k in the paper
    # @NLexpression(CCUS, TempTerm[s in 1:S, k in 1:K], (4/pi)*8.314*temp_s[s]*(sum(T_s_k_t[s,k,t] for s in 1:S, t in 1:T) + sum(U_s_k[s,k] for s in 1:S)))
    @NLexpression(CCUS, SqrtTerm[s in 1:S, k in 1:K], sum(T_s_k_t[s,k,t] for s in 1:S, t in 1:T) + sum(U_s_k[s,k] for s in 1:S) ) #######should have the sqrt...
    #####    ##   
    ##   ##  #### 
    ##       ##  ## 
    ##      ##    ## 
    ##      ########  
    ##   ## ##    ##  Not using the sqrt in D_s_k bacuase of error...
    #####  ##    ## 
    DconstMatrix = zeros(S,K)
    for s in 1:S, k in 1:K
        DconstMatrix[s,k] = sqrt(velocity[s,k]*Mass_s[s]*(PressureDiffMatrix[s,k] + PressDropPara_s_k[s,k]))
    end
    # @NLexpression(CCUS, DenomTerm[s in 1:S, k in 1:K], sqrt(velocity[s,k]*Mass_s[s]*(PressureDiffMatrix[s,k] + PressDropPara_s_k[s,k])))
    # @NLexpression(CCUS, D_s_k[s in 1:S, k in 1:K], sqrt(TempTerm[s,k]) / DconstMatrix[s,k] )
    @NLexpression(CCUS, D_s_k[s in 1:S, k in 1:K], (eps_nl+(4/pi)*8.314*temp_s[s]*SqrtTerm[s,k] / DconstMatrix[s,k])^0.5 )
    # @expression(CCUS,PipeCost[s in 1:S,k in 1:K], (95230*round(D_s_k[s,k]) + 96904)*CRF)#with the round
    @NLexpression(CCUS,PipeCost[s in 1:S,k in 1:K], (95230*D_s_k[s,k] + 96904)*CRF)#with the round

    @NLexpression(CCUS,TranspCost, sum(DistSourSink_s_k[s,k]*PipeCost[s,k] for s in 1:S, k in 1:K))
    @expression(CCUS,SinkCost, sum(F_k[k]*CR_sink_k[k] for k in 1:K))
    @NLexpression(CCUS, TAC,TreatCost + compressionCost + TranspCost + SinkCost) #The cost of compression is determined by considering both compressor and pump costs

    #emission:
    @expression(CCUS,SourceEmiss, sum((1-theta_s[s])*R_s[s] for s in 1:S))
    @expression(CCUS,TreatEmiss[t in 1:T],sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for s in 1:S,k in 1:K))
    @expression(CCUS, TranspEmiss,TransportPow*epsilon_p)
    @expression(CCUS, SinkEmiss, sum(F_k[k]*Mu_k[k] for k in 1:K))
    @expression(CCUS,TotEmiss,SourceEmiss+sum(TreatEmiss)+TranspEmiss+SinkEmiss)#Emission objective
    return (m=CCUS, TAC=TAC, TotEmiss=TotEmiss, InherSafeIndic=InherSafeIndic)
end

# Convenience: 2-objective simplex weights (returns Vector{NTuple{2,Float64}}).
# Each pair (w1,w2) with w1=a/K, w2=1-w1 for a=0..K.
function simplex_weights2(K::Integer)
    W2 = NTuple{2,Float64}[]
    for a in 0:K
        w1 = a / K
        w2 = 1.0 - w1
        push!(W2, (w1, w2))
    end
    return W2
end

# Example: use `W = simplex_weights2(50)` for two-objective runs.

W = simplex_weights2(300)
println("Total weight combinations: ", W)
all_points = NamedTuple[]

for (i, j) in W
    bm = build_model()
    objTAC  = bm.TAC
    Emiss  = bm.TotEmiss
    ISI = bm.InherSafeIndic
    println("Weights: ISI + Emiss=", i, "TAC=", j)
    @NLobjective(bm.m, Min, i*(ISI*500000+Emiss*6)+ j*objTAC)
    optimize!(bm.m)
    push!(all_points, (TAC=value(bm.TAC), Emiss=value(bm.TotEmiss), ISI=value(bm.InherSafeIndic)))
end

if !isempty(all_points)
    println("--- All Feasible Points Found (Before Filtering) ---")
    println(DataFrame(all_points))
    println("----------------------------------------------------")
end

df_all = DataFrame(all_points)
## 3d plots
p3d = scatter(
    df_all.Emiss, df_all.ISI, df_all.TAC,
    xlabel="Emission", ylabel="ISI", zlabel="TAC",
    title="3D Pareto Frontier",
    legend=false,
    markersize=4, markerstrokewidth=0
)
display(p3d)

p2d_all = scatter(
    df_all.ISI, df_all.Emiss,
    marker_z=df_all.TAC,
    colorbar=true,
    colorbar_title="TAC",
    xlabel="ISI",
    ylabel="Emission",
    # title="2D Projection of Pareto Frontier (Color by TAC)",
    legend=false,
    markersize=5
)
display(p2d_all)
p2d_Pareto = scatter(
    df_all.ISI*5000000+6*df_all.Emiss, df_all.TAC,
    xlabel="ISI+Emission",
    ylabel="TAC",
    title="Objective Reduction Pareto Frontier",
    legend=false,
    markersize=5
)
display(p2d_Pareto)