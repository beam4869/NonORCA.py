using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX,LinearAlgebra,Ipopt, BARON

S = 4 #Number of different carbon sources Ammonia, Steel, Refinery, Power Plant.  from Table 3
R_s = [357, 1260, 399, 3426] #Raw carbon source flow from s
y_s = [1, 0.44,0.27,0.07] # CO2 composition from Table 3.
y_s_t = [1,0.99,0.99,0.99] #the purity of the produced CO2 is over 99 %
K = 6 # NUmber of different carbon sinks
Zmin_k = [0.06,0.94,0.94,0.999,0.999,0.999]
Gmax_k = [126,508,3168,756,543,913] # max CO2 flow rate.
L_s_k = ones(S,K)*30 # min for T+U madeup number
M_s_k = ones(S,K)*3100# max for T+U
Mu_k = [0.42,0.5,0.098,0.39,0,0] #sinks efficiency
ComprePowPara_s_k = [0.43 2.22 4.37 4.37 4.37 4.37;0.49 2.23 4.37 4.37 4.37 4.37;0.62 2.28 4.37 4.37 4.37 4.37;0.58 2.28 4.37 4.37 4.37 4.37] #compressor power CO2 parameter kW d/t CO2 from 2016 Al-Mohannadi.
PumpPowPara_s_k = [0 0 0.204 0.017 0.176 0.204;0 0 0.204 0.017 0.176 0.204;0 0 0.205 0.018 0.176 0.203;0 0 0.205 0.018 0.175 0.203] #Pump power CO2 parameter kW d/t CO2 from 2016 Al-Mohannadi.
PressDropPara_s_k = [56.39 832.13 49.51 49.51 50.82 51.15;67.87 843.61 60.98 60.98 62.30 62.62; 90.82 896.07 96.72 96.72 29.84 16.72;82.95 888.20 88.85 88.85 21.64 26.89]
DistSourSink_s_k = [1.72 25.38 1.51 1.51 1.55 1.56;2.07 25.73 1.86 1.86 1.9 1.91;2.77 27.33 2.95 2.95 0.91 0.51;2.53 27.09 2.71 2.71 0.66 0.82]

T = 1 # NUmbers of different treatment technologies. there is only one in the literature
R = 1 # NUmber of different transportation options.
#cost parameters
epsilon_t = [0.9] # The capture efficiency of the amine capturing unit is assumed to be 90 %
epsilon_p = 0.366 #power carbon foot print kg CO2/kwh
powerprice = 0.02 #power price USD/KWh price in Qatar, it's much lower than in the USA
CRF = 0.15 #capital recovery factor
gamma_t = [0.0338] # The secondary carbon emissions parameter γt for the treatment unit was assumed to be 0.0338 tons of CO2 produced per ton of CO2 processed
CapturePercent = 0.4 # hyperparameters
TranspCost_s_t = [0,29,43,35]
#Safety parameters
I_tot_r = [12] #total safety index of transportation from Table 5 
I_tot_t = [19] #total safety index of treatment from Table 5 
I_tot_k = [7,7,8,36,25,34] #total safety index of sinks from Table 5 
temp_s = [298,298,298,298] #temperature in the source (K)
SourcePress_s = 101*ones(S) # pressure in the source
SinkPress_k = [101,101,8080,14140,15198,15198]
PressureDiffMatrix = zeros(S,K) # pressure difference between sink and source
for s in 1:S
  for k in 1:K
    PressureDiffMatrix[s,k] = SinkPress_k[k] - SourcePress_s[s]
  end
end 
velocity = 20*ones(S,K) #outlet velocity from sources (assumed at 20 m/s)
Mass_s = 44*ones(S) #molecular weight
DconstMatrix = zeros(S,K)
for s in 1:S, k in 1:K
  DconstMatrix[s,k] = sqrt(velocity[s,k]*Mass_s[s]*(PressureDiffMatrix[s,k] + PressDropPara_s_k[s,k]))
end
CR_sink_k = [-7,-5,9,-20,-17,-26] # processing cost parameters (CRsink )
theta_s = 0.99*ones(S)


CCUS = Model(CPLEX.Optimizer)
########varaiables
@variable(CCUS,T_s_k_t[1:S,1:K,1:T] >=0) # treated carbon flow
@variable(CCUS,U_s_k[1:S,1:K] >=0) # carbon flow without treatment
@variable(CCUS,F_k[1:S,1:K] >=0) # total flow into any sink
@variable(CCUS,X_s_k[1:S,1:K],Bin) # total flow into any sink
# @variable(CCUS,P_k[1:K]>=0) # power usage in sinks


@constraint(CCUS,totMassBalance[s in 1:S], R_s[s] == sum(T_s_k_t[s,k,t] for t in 1:T, k in 1:K) + sum(U_s_k[s,k] for k in 1:K) ) # total mass balance
# @constraint(CCUS,carbonMassBalance[s in 1:S], R_s[s]*y_s[s] == sum(sum(epsilon_t[t]*T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T) for k in 1:K) + sum(U_s_k[s,k]*y_s[s] for k in 1:K) ) # carbon mass balance
# @constraint(CCUS,carbonMassBalance[s in 1:S], R_s[s]*y_s[s] == sum(T_s_k_t[s,k,t]*y_s[s]*y_s_t[s,t]*epsilon_t[t] for t in 1:T, k in 1:K) + sum(U_s_k[s,k]*y_s[s] for k in 1:K) ) # T should divide the epsilon!

@constraint(CCUS,sinkTotMassBalance[k in 1:K],F_k[k] == sum(sum(T_s_k_t[s,k,t] for t in 1:T) for s in 1:S)+sum(U_s_k[s,k] for s in 1:S)) 
@constraint(CCUS,sinkMinConcenRequ[k in 1:K],F_k[k]*Zmin_k[k] <= sum(sum(T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T) for s in 1:S)+sum(U_s_k[s,k]*y_s[s] for s in 1:S)) 
@constraint(CCUS,MaxSinkFlow[k in 1:K], F_k[k]<=Gmax_k[k])
# @constraint(CCUS,MinTandUFlow[s in 1:S,k in 1:K,t in 1:T], L_s_k[s,k]*X_s_k[s,k]<=T_s_k_t[s,k,t]+U_s_k[s,k]) #min for the total flow from s to k
# @constraint(CCUS,MaxTandUFlow[s in 1:S,k in 1:K,t in 1:T], M_s_k[s,k]*X_s_k[s,k]>=T_s_k_t[s,k,t]+U_s_k[s,k]) #max for the total flow from s to k

####Transport cost
@expression(CCUS, CompressionPow,24*sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T) ) # compressor power. per year 24*
@expression(CCUS, PumpPow,24*0.8*sum(PumpPowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T) ) # sum of the pump power and compressor power. per year
@expression(CCUS, TransportPow, CompressionPow + PumpPow ) # sum of the pump power and compressor power. per year
##emission expression
@expression(CCUS,SourceEmiss, sum((1-theta_s[s])*R_s[s] for s in 1:S))
@expression(CCUS,TreatEmiss[t in 1:T],sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for s in 1:T,k in 1:K))
@expression(CCUS, TranspEmiss,TransportPow*epsilon_p)
@expression(CCUS, SinkEmiss, sum(F_k[k]*Mu_k[k] for k in 1:K))
@expression(CCUS,TotEmiss,SourceEmiss+sum(TreatEmiss)+TranspEmiss+SinkEmiss)#Emission objective

@objective(CCUS,Min,TotEmiss)
JuMP.optimize!(CCUS)
println("Emission value:", value(TotEmiss))
println("Treated carbon flow:", value.(T_s_k_t))
println("Non-Treated carbon flow:", value.(U_s_k))