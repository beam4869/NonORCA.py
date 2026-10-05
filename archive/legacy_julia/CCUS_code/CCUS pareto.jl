#construct a sample CCUS supply chain model refering the paper Safety-driven design 
#of carbon capture utilization and storage (CCUS) supply chains: A multi-objective optimization approach
#finite difference 
using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX,LinearAlgebra,Ipopt, BARON, Plots

S = 4 #Number of different carbon sources Ammonia, Steel, Refinery, Power Plant.  from Table 3
M_s = [357, 1260, 399, 3426] #Raw carbon source flow from s
L_s=zeros(S)
y_s = [1, 0.44,0.27,0.07] # CO2 composition from Table 3.
CO2_gen=sum(M_s.*y_s)
y_s_t = [1,0.999,0.999,0.999] #the purity of the produced CO2 is over 99 %
K = 6 # NUmber of different carbon sinks
Zmin_k = [0.06,0.94,0.94,0.999,0.999,0.999]
Gmax_k = [126,508,3168,756,543,913] # max CO2 flow rate.
#Gmax_k = [10^6,10^6,10^6,10^6,10^6,10^6] # max CO2 flow rate.
L_s_k = zeros(S,K)#[10 10 3 10 30 10; 80 150 70 80 200 50;35 30 25 30 20 20;500 400 650 600 200 350] # min for T+U madeup number
M_s_k = [126 508 3168 756 543 913; 126 508 3168 756 543 913; 126 508 3168 756 543 913; 126 508 3168 756 543 913] # max for T+U
Mu_k = [0.28,0.5,0,0.17,0.11,0.34] #sinks efficiency (using safety paper values)
#Mu_k = [.42,.5,0,.098,.39,0] #Sinks efficiency using industrial park paper values
ComprePowPara_s_k = [0.43 2.22 4.37 4.37 4.37 4.37;0.49 2.23 4.37 4.37 4.37 4.37;0.62 2.28 4.37 4.37 4.37 4.37;0.58 2.28 4.37 4.37 4.37 4.37] #compressor power CO2 parameter kW d/t CO2 from 2016 Al-Mohannadi.
PumpPowPara_s_k = [0 0 0.204 0.017 0.176 0.204;0 0 0.204 0.017 0.176 0.204;0 0 0.205 0.018 0.176 0.203;0 0 0.205 0.018 0.175 0.203] #Pump power CO2 parameter kW d/t CO2 from 2016 Al-Mohannadi.
PressDropPara_s_k = [56.39 832.13 49.51 49.51 50.82 51.15;67.87 843.61 60.98 60.98 62.30 62.62; 90.82 896.07 96.72 96.72 29.84 16.72;82.95 888.20 88.85 88.85 21.64 26.89]
DistSourSink_s_k = [1.72 25.38 1.51 1.51 1.55 1.56;2.07 25.73 1.86 1.86 1.9 1.91;2.77 27.33 2.95 2.95 0.91 0.51;2.53 27.09 2.71 2.71 0.66 0.82]

T = 1 # NUmbers of different treatment technologies. there is only one in the literature
R = 1 # NUmber of different transportation options.
#cost parameters
epsilon_t = [0.9] # The capture efficiency of the amine capturing unit is assumed to be 90 %
# epsilon_p = 0.366 #power carbon foot print kg CO2/kwh
epsilon_p = 0.366 #power carbon foot print kg CO2/kwh
powerprice = 0.02 #power price USD/KWh price in Qatar, it's much lower than in the USA
CRF = 0.15 #capital recovery factor
gamma_t = [0.0338] # The secondary carbon emissions parameter γt for the treatment unit was assumed to be 0.0338 tons of CO2 produced per ton of CO2 processed
CapturePercent = 0.4 # hyperparameters value from the paper.
#CapturePercent = 0.21# hyperparameters
TranspCost_s_t = [0,29,43,35]
#Safety parameters
I_tot_r = [12] #total safety index of transportation from Table 5 
I_tot_t = [19] #total safety index of treatment from Table 5 
I_tot_k = [7,7,8,36,25,34] #total safety index of sinks from Table 5 

itera = 10
step = 1/(itera)
ParetoMatrix = zeros(itera,3)
for i in 1:itera
    # for j in 1:itera+1-i
# CCUS = Model(Ipopt.Optimizer)
CCUS = Model(BARON.Optimizer)
set_optimizer_attribute(CCUS,"maxtime",60)

########varaiables
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
#@constraint(CCUS,MaxTandUFlow[s in 1:S,k in 1:K,t in 1:T], M_s_k[s,k]*X_s_k[s,k]>=T_s_k_t[s,k,t]+U_s_k[s,k]) #max for the total flow from s to k
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
@NLexpression(CCUS, InherSafeIndic, sum(I_tot_k[k]*(sum(T_s_k_t[s,k,t] for s in 1:S,t in 1:T)+sum(U_s_k[s,k] for s in 1:S))/Gmax_k[k] for k in 1:K)+sum(I_tot_r[r]*sum(R_s[s] for s in 1:S)/sum(M_s[s] for s in 1:S) for r in 1:R) + sum(I_tot_t[t]*sum(T_s_k_t[s,k,t]+V_s_k_t[s,k,t] for s in 1:S,k in 1:K)/sum(M_s[s] for s in 1:S) for t in 1:T))

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
temp_s = [298,298,298,298] #temperature in the source (K)
SourcePress_s = 101*ones(S) # pressure in the source
SinkPress_k = [101,101,8080,14140,15198,15198]
PressureDiffMatrix = zeros(S,K) # pressure difference between sink and source
for s in 1:S
  for k in 1:K
    PressureDiffMatrix[s,k] = SinkPress_k[k] - SourcePress_s[s]
  end
end 
println("pressure:",PressureDiffMatrix)
velocity = 20*ones(S,K) #outlet velocity from sources (assumed at 20 m/s)
Mass_s = 44*ones(S) #molecular weight
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
@NLexpression(CCUS, D_s_k[s in 1:S, k in 1:K], ((4/pi)*8.314*temp_s[s]*SqrtTerm[s,k] / DconstMatrix[s,k])^0.5 )
# @expression(CCUS,PipeCost[s in 1:S,k in 1:K], (95230*round(D_s_k[s,k]) + 96904)*CRF)#with the round
@NLexpression(CCUS,PipeCost[s in 1:S,k in 1:K], (95230*D_s_k[s,k] + 96904)*CRF)#with the round

@NLexpression(CCUS,TranspCost, sum(DistSourSink_s_k[s,k]*PipeCost[s,k] for s in 1:S, k in 1:K))
CR_sink_k = [-7,-5,9,-20,-17,-26] # processing cost parameters (CRsink )
@expression(CCUS,SinkCost, sum(F_k[k]*CR_sink_k[k] for k in 1:K))
@NLexpression(CCUS, TAC,TreatCost + compressionCost + TranspCost + SinkCost) #The cost of compression is determined by considering both compressor and pump costs

#emission:
theta_s = 0.99*ones(S)
@expression(CCUS,SourceEmiss, sum((1-theta_s[s])*R_s[s] for s in 1:S))
@expression(CCUS,TreatEmiss[t in 1:T],sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for s in 1:S,k in 1:K))
@expression(CCUS, TranspEmiss,TransportPow*epsilon_p)
@expression(CCUS, SinkEmiss, sum(F_k[k]*Mu_k[k] for k in 1:K))
@expression(CCUS,TotEmiss,SourceEmiss+sum(TreatEmiss)+TranspEmiss+SinkEmiss)#Emission objective


println("I am in 280")

##################ISI objective

# @NLobjective(CCUS,Min,InherSafeIndic)
# @NLobjective(CCUS,Min,TAC)
# @NLobjective(CCUS,Min,TotEmiss)

# @NLobjective(CCUS,Max,NetCapture)
# JuMP.optimize!(CCUS)

# @NLobjective(CCUS,Min,step*i*TAC+step*j*TotEmiss+(1-step*i-step*j)*InherSafeIndic) #step is the hyperparameter to balance the objectives
# @NLobjective(CCUS,Min,step*(i-1)*TotEmiss+(1-step*(i-1))*InherSafeIndic*100000) #step is the hyperparameter to balance the objectives
@NLobjective(CCUS,Min,step*(i-1)*InherSafeIndic*30000+(1-step*(i-1))*TotEmiss) #step is the hyperparameter to balance the objectives
JuMP.optimize!(CCUS)
ParetoMatrix[i,1] = value(TAC)
ParetoMatrix[i,2] = value(TotEmiss)
ParetoMatrix[i,3] = value(InherSafeIndic)
# end
end
# display(scatter(ParetoMatrix[:,2],ParetoMatrix[:,3],xlabel = "Total emission", ylabel = "ISI",size=[800,250],margin=10Plots.mm))
# display(scatter(ParetoMatrix[:,1],ParetoMatrix[:,3],xlabel = "Total cost", ylabel = "ISI",size=[800,250],margin=10Plots.mm))
display(scatter(ParetoMatrix[:,3],ParetoMatrix[:,2],xlabel = "Total ISI", ylabel = "Total Emiss",size=[800,250],margin=10Plots.mm))
# scatter3d(ParetoMatrix[:, 1], ParetoMatrix[:, 2], ParetoMatrix[:, 3],
#     xlabel = "Total Costs",
#     ylabel = "Total Emissions",
#     zlabel = "Inherent Safety Indicator",
#     title = "3D Pareto Frontier",
#     marker = (:circle, 8),
#     color = :blue)
# println("ISI value:", value(InherSafeIndic))
# println("Net Capture value", value(NetCapture))
# println("F CO2 value: ", value.(F_CO2_k))
