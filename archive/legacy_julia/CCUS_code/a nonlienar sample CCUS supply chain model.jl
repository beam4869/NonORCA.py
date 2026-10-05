#construct a sample CCUS supply chain model refering the paper Safety-driven design 
#of carbon capture utilization and storage (CCUS) supply chains: A multi-objective optimization approach
#finite difference 
using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX,LinearAlgebra,Ipopt, BARON
function reduced(A,red)
    for i in 1:size(A,1)
        for j in 1:size(A,1)
            A[i,j]=round(A[i,j]*1000,digits=0) # make the numbers be round in the first digit
        end
    end
    #println(A)
    global res=1
    global ov_ccom=Leiden.leiden(A,resolution=res)
    global ov_cpart=ov_ccom[2] # the partition vector
    #run leiden until resolution gives no communities
    count = 0
    while length(ov_cpart)!=size(A,1)
        # count += 1
        global ov_ccom=Leiden.leiden(A,resolution=res)
        global ov_cpart=ov_ccom[2]
        global res+=1 #when all the elements are divided individually, loop stops.
        
    end   # but at this time the value of res is actually one larger than the res that all elements are divided individually
    #run leiden until "red" number of objective communities is reached
    count = 0
    while ( length(ov_cpart)>red && res>0) #
        # count += 1
        global ov_ccom=Leiden.leiden(A,resolution=res)
        global ov_cpart=ov_ccom[2]
        global res-=.1
    end

    return(ov_cpart) # just give out a value that has less dimension than red, but I think that just should be the integer lower than red, can't figure out the meaning of it...
end  
function NLPCorrStrengGenerating(IneqMatrix,ObjMatrix,EqMatrix,NoParitition,NoOfSP) # version 8/2/2024 consider the condition that EqMatrix is all zero
    # IneqMatrix, Jacobian matrix of inequal constaints; ObjMatrix, Jacobian matrix of objectives; EqMatrix, Jacobian matrix of equal constaints; NoParitition and NoOfSP, No. of paritions and selected points
    NoOfObj = Int(size(ObjMatrix,1)/NoOfSP) # number of selected points times number of obectives equals the numbers objectives in the ObjMatrix
    # println("NoOfObj = ", NoOfObj)
    normc=zeros(size(ObjMatrix)) # normalized Cmat matrix
    for i in 1:size(ObjMatrix,1)
      normc[i,:]=-normalize(ObjMatrix[i,:])#normalize the value of the matrix row by row
    end # thus normc is the normalized ce
    posmin=0.9 # hyperparameters in weight function usually 0.9
    beta = 100 # hyperparameters in weight function usually 100
    IneqSecstrength=zeros(NoOfObj,NoOfObj,size(IneqMatrix,1),NoOfSP) #Sijkn here for inequal constraints
    totIneqWt=zeros(NoOfObj,NoOfObj,size(IneqMatrix,1),NoOfSP) #Wijkn here for inequal constraints
    #Inequality constraints
    for i in 1:NoOfObj
      for j in i+1:NoOfObj
        for k in 1:size(IneqMatrix,1)
          for n in 1:NoOfSP
            # println("principle one: ", dot(normc[i,:],Ae[k,:]))
            # println("principle two: ", dot(normc[j,:],Ae[k,:]))
            cp1=normc[(i-1)*NoOfSP+n,:]-dot(normc[(i-1)*NoOfSP+n,:],IneqMatrix[k,:])/(sqrt(sum(IneqMatrix[k,p]^2 for p in 1:size(IneqMatrix,2)))^2)*IneqMatrix[k,:] # to project normc[i] on Ae[K]
            cn1=normc[(i-1)*NoOfSP+n,:]-cp1 # get the left part that is normal to Ae[k]
            cp2=normc[(j-1)*NoOfSP+n,:]-dot(normc[(j-1)*NoOfSP+n,:],IneqMatrix[k,:])/(sqrt(sum(IneqMatrix[k,p]^2 for p in 1:size(IneqMatrix,2)))^2)*IneqMatrix[k,:]
            cn2=normc[(j-1)*NoOfSP+n,:]-cp2
            if cp1!=zeros(length(cp1)) 
                cp1norm=normalize(cp1) # normalize the proejcted vectors
            else
                cp1norm=cp1 # function narmalize can be applied to all zeros!!!
            end
            if cp2!=zeros(length(cp2))
                cp2norm=normalize(cp2)
            else
                cp2norm=cp2
            end
            if dot(normc[(i-1)*NoOfSP+n,:],IneqMatrix[k,:])!=0 && dot(normc[(j-1)*NoOfSP+n,:],IneqMatrix[k,:])!=0 # if this two objective vectors are not normal to a row in the constrain matrix
              #Previously we don't count the objecives that are pointing into the contraints because we can only have optimal solutions in the boundary of the feasible region in linear problem.
              #But now we are considering nonlinear problem, we need to consider all the objectives. Optimal points can be in the middle of the feasible region when objectives are nonliear.
              if dot(IneqMatrix[k,:],cn1)>0 && dot(IneqMatrix[k,:],cn2)>0 # normally the dot product results should be zero because the cni is normal to Ae[k]
                IneqSecstrength[i,j,k,n] = dot(cp1norm,cp2norm)
                totIneqWt[i,j,k,n] = 1-posmin*(1/(1+exp(-beta*IneqSecstrength[i,j,k,n]))) # calculat the weight Sij*Wijk
                # if secstrength==0
                #   totIneqWt[i,j,k]=0 # why
                # else
                #   totIneqWt[i,j,k]+=(1-posmin*(1/(1+exp(-beta*secstrength)))) # store the Wijk
                # end
              else
                # println("one here")siz
                IneqSecstrength[i,j,k,n]=0
                totIneqWt[i,j,k,n] = 0
              end
            else#if one objective is perellel to the constraint, I think we should still calculate it becuase its also in the boundary and there should be tradeoffs.
              IneqSecstrength[i,j,k,n] = dot(cp1norm,cp2norm)
              totIneqWt[i,j,k,n] = 1-posmin*(1/(1+exp(-beta*IneqSecstrength[i,j,k,n])))
            end
            #println(cp1,cp2)
            #println(cp1norm, cp2norm)
            #println(secstrength, " secondary strength")
            #dsm[i,j]+=(1-(secstrength+1)*0.5)*secstrength#secstrength
          end
        end
      end
    end
    ##############################################################################
    #Equality constraints
    EqSecstrength=zeros(NoOfObj,NoOfObj,size(EqMatrix,1),NoOfSP) #Sijkn here for equal constraints
    totEqWt=zeros(NoOfObj,NoOfObj,size(EqMatrix,1),NoOfSP) #Wijkn here for equal constraints
    for i in 1:NoOfObj
      for j in i+1:NoOfObj
        for k in 1:size(EqMatrix,1)
          for n in 1:NoOfSP
            if sum(EqMatrix) !=0 # if there is no equal constraint in the problem, we just make the weights are zero. 
              #Also there will be NaN in the cps if this if statement is not added..
              cp1=normc[(i-1)*NoOfSP+n,:]-dot(normc[(i-1)*NoOfSP+n,:],EqMatrix[k,:])/(sqrt(sum(EqMatrix[k,p]^2 for p in 1:size(EqMatrix,2)))^2)*EqMatrix[k,:]
              cp2=normc[(j-1)*NoOfSP+n,:]-dot(normc[(j-1)*NoOfSP+n,:],EqMatrix[k,:])/(sqrt(sum(EqMatrix[k,p]^2 for p in 1:size(EqMatrix,2)))^2)*EqMatrix[k,:]
      
              if cp1!=zeros(length(cp1))
                  cp1norm=normalize(cp1)
              else
                  cp1norm=cp1
              end
              if cp2!=zeros(length(cp2))
                  cp2norm=normalize(cp2)
              else
                  cp2norm=cp2
              end
              if dot(normc[(i-1)*NoOfSP+n,:],EqMatrix[k,:])!=0 && dot(normc[(j-1)*NoOfSP+n,:],EqMatrix[k,:])!=0
                EqSecstrength[i,j,k,n] = dot(cp1norm,cp2norm)
                totEqWt[i,j,k,n] = 1
                #   if seceqstren==0
                #       toteqwt[i,j,k]=0
                #   else
                #       toteqwt[i,j,k]+=(1-posmin*(1/(1+exp(-beta*seceqstren)))) # might be a bug
                #   end
                # println("Wijk: ", toteqwt[i,j,k]) # No abnorm
              else
                EqSecstrength[i,j,k,n] = dot(cp1norm,cp2norm)
                totEqWt[i,j,k,n] = 1
              end
            else
              EqSecstrength[i,j,k,n] = 0
              totEqWt[i,j,k,n] = 0
            end
          end
        end
      end
    end
    totwt= zeros(NoOfObj,NoOfObj)
    totSecWt=zeros(NoOfObj,NoOfObj)
    #deltaine=ones(size(ce,1),size(ce,1),size(Ae,1))
    #deltaeq=ones(size(ce,1),size(ce,1),size(de,1))
    for i in 1:NoOfObj
      for j in i+1:NoOfObj
        totwt[i,j]=totwt[j,i]= sum(totIneqWt[i,j,:,:])+sum(totEqWt[i,j,:,:])#sum of the total weight 
        totSecWt[i,j]=totSecWt[j,i]+=sum(totIneqWt[i,j,:,:].*IneqSecstrength[i,j,:,:])+sum(totEqWt[i,j,:,:].*EqSecstrength[i,j,:,:])
      end
    end
    adjMatrix=zeros(NoOfObj,NoOfObj)
    combmin=zeros(NoOfObj,NoOfObj)
    for i in 1:NoOfObj
      for j in i+1:NoOfObj
        adjMatrix[i,j]=adjMatrix[j,i]=(1/2)*(1 + totSecWt[i,j]/totwt[i,j])
        # combmin[i,j]=combmin[j,i]=minimum(vcat(dsmeq[i,j,:],dsm[i,j,:])) #debuging line.
      end
    end
    # for i in 1:NoOfObj
    #     for j in i+1:NoOfObj
    #         if isnan(comb[i,j])
    #             comb[i,j]=comb[j,i]=0
    #         end
    #     end
    # end
    # for i in 1:NoOfObj
    #     for j in i+1:NoOfObj
    #         comb[i,j]=comb[j,i]=(1/2)*(comb[i,j]+1)# result of Sij
    #         # combmin[i,j]=combmin[j,i]=(1/2)*(combmin[i,j]+1)
    #     end
    # end
    for i in 1:NoOfObj
      adjMatrix[i,i] = 1
    end
    println(adjMatrix)
    groups=reduced(copy(adjMatrix),NoParitition)
    
    #dpm, dsmfin, dsmeqfin, groups,
    return adjMatrix, totwt, groups , totSecWt, EqSecstrength, totEqWt, IneqSecstrength,totIneqWt#, dsmfin, dsmeqfin, dpm
end 
#Initialize the convex optimization problem and obtain the selected points by iteration.
function selectedPointGeneration(delta,xn) 
  #optimization model
  original = Model(Ipopt.Optimizer)
  @variable(original, x[1:n])
  # @constraint(original, CircleCovnstraint, x[1]^2 + x[2]^2/4+x[3]^2/9 <= 1) 
  @constraint(original, xConstraints1[i in 1:n], x[i] >= 0) 
  @constraint(original, xConstraints2[i in 1:n], x[i] <= 1) 
  #objective
  @objective(original, Min, sum((x[i] - xn[i] - delta[i] )^2 for i in 1:DimNum))
  JuMP.optimize!(original) # depend on the package that you are using.
  return value.(x)#, original
end
function deltaGeneration(Ae,ce,de) # Ae unequal constraints. Ce objectives. De equal constraints. 
  normc=zeros(size(ce)) # normalized Cmat matrix
  delta  = zeros(size(ce,2))
  for i in 1:size(ce,1)
    normc[i,:]=-normalize(ce[i,:])#normalize the value of the matrix row by row
  end # thus normc is the normalized ce
  omega = rand(Float64, size(ce,1))
  # if there is no eqaul constraints, there will be NaN of the division process, therefore, calculate it with two different conditions.
  if sum(de) != 0 
    cpik = zeros(size(ce,1),size(Ae,1)+size(de,1),size(ce,2))
    for i in 1:size(ce,1)
      for j in 1:size(Ae,1)
        cpik[i,j,:] = normc[i,:]-dot(normc[i,:],Ae[j,:])/(sqrt(sum(Ae[j,p]^2 for p in 1:size(Ae,2)))^2)*Ae[j,:] # to project normc[i] on Ae[K]
      end
      for j in size(Ae,1)+1:size(Ae,1)+size(de,1)
        cpik[i,j,:] = normc[i,:]-dot(normc[i,:],de[j-size(Ae,1),:])/(sqrt(sum(de[j-size(Ae,1),p]^2 for p in 1:size(de,2)))^2)*de[j-size(Ae,1),:] # to project normc[i] on de[K]
      end
    end
    delta = sum(sum(omega[i]*cpik[i,k,:] for i in 1:size(ce,1))/sum(omega[i] for i in 1:size(ce,1)) for k in 1:size(Ae,1)+size(de,1))/(size(Ae,1)+size(de,1))

  else
    cpik = zeros(size(ce,1),size(Ae,1),size(ce,2))
    for i in 1:size(ce,1)
      for j in 1:size(Ae,1)
        cpik[i,j,:] = normc[i,:]-dot(normc[i,:],Ae[j,:])/(sqrt(sum(Ae[j,p]^2 for p in 1:size(Ae,2)))^2)*Ae[j,:] # to project normc[i] on Ae[K]
      end
    end
    delta = sum(sum(omega[i]*cpik[i,k,:] for i in 1:size(ce,1))/sum(omega[i] for i in 1:size(ce,1)) for k in 1:size(Ae,1))/(size(Ae,1))
  end
  return delta
end
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


# CCUS = Model(Ipopt.Optimizer)
CCUS = Model(BARON.Optimizer)
########varaiables
@variable(CCUS,T_s_k_t[1:S,1:K,1:T] >=0) # treated carbon flow
@variable(CCUS,U_s_k[1:S,1:K] >=0) # carbon flow without treatment
@variable(CCUS,F_k[1:S,1:K] >=0) # total flow into any sink
@variable(CCUS,X_s_k[1:S,1:K],Bin) # total flow into any sink
# @variable(CCUS,P_k[1:K]>=0) # power usage in sinks

@constraint(CCUS,totMassBalance[s in 1:S], R_s[s] == sum(T_s_k_t[s,k,t] for t in 1:T, k in 1:K) + sum(U_s_k[s,k] for k in 1:K) ) # total mass balance
@constraint(CCUS,carbonMassBalance[s in 1:S], R_s[s]*y_s[s] == sum(sum(epsilon_t[t]*T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T) for k in 1:K) + sum(U_s_k[s,k]*y_s[s] for k in 1:K) ) # carbon mass balance
##total and carbon mass balance for sinks:
@constraint(CCUS,sinkTotMassBalance[k in 1:K],F_k[k] == sum(sum(T_s_k_t[s,k,t] for t in 1:T) for s in 1:S)+sum(U_s_k[s,k] for s in 1:S)) 
@constraint(CCUS,sinkMinConcenRequ[k in 1:K],F_k[k]*Zmin_k[k] <= sum(sum(T_s_k_t[s,k,t]*y_s_t[s,t] for t in 1:T) for s in 1:S)+sum(U_s_k[s,k]*y_s[s] for s in 1:S)) 
@constraint(CCUS,MaxSinkFlow[k in 1:K], F_k[k]<=Gmax_k[k])
@constraint(CCUS,MinTandUFlow[s in 1:S,k in 1:K,t in 1:T], L_s_k[s,k]*X_s_k[s,k]<=T_s_k_t[s,k,t]+U_s_k[s,k]) #min for the total flow from s to k
@constraint(CCUS,MaxTandUFlow[s in 1:S,k in 1:K,t in 1:T], M_s_k[s,k]*X_s_k[s,k]>=T_s_k_t[s,k,t]+U_s_k[s,k]) #max for the total flow from s to k

@expression(CCUS, CompressionPow,24*sum(ComprePowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T) ) # compressor power. per year 24*
@expression(CCUS, PumpPow,24*0.8*sum(PumpPowPara_s_k[s,k]*(T_s_k_t[s,k,t]+U_s_k[s,k]) for s in 1:S, k in 1:K, t in 1:T) ) # sum of the pump power and compressor power. per year
@expression(CCUS, TransportPow, CompressionPow + PumpPow ) # sum of the pump power and compressor power. per year

@expression(CCUS, NetCapture, sum(F_k[k]*(1-Mu_k[k]) for k in 1:K) - sum( sum(sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for t in 1:T) for k in 1:K) for s in 1:S) - TransportPow*epsilon_p ) 
@constraint(CCUS,CarbonCapGoal, NetCapture >= 14.905*356*CapturePercent) # want the total carbon capture satisfy the set goal from Section 3 A daily capacity of 14,905 t of CO2 is available for capture, utilization, storage, or controlled emission.
#safety objective
@NLexpression(CCUS,ISISqrtExpre1,sum(T_s_k_t[s,k,t]^2 for s in 1:S, k in 1:K, t in 1:T)^0.5 )
@NLexpression(CCUS,ISISqrtExpre2,sum(U_s_k[s,k]^2 for s in 1:S, k in 1:K)^0.5 )
# @expression(CCUS, InherSafeIndic, sum( sum((I_tot_t[t]+I_tot_r[r]+I_tot_k[k])*normalize(T_s_k_t)[s,k,t] for t in 1:T, r in 1:R) + sum((I_tot_r[r]+I_tot_k[k])*normalize(U_s_k)[s,k] for r in 1:R) for s in 1:S, k in 1:K)) #equation 15 and 16
# @NLexpression(CCUS, InherSafeIndic, sum( sum((I_tot_t[t]+I_tot_r[r]+I_tot_k[k])*T_s_k_t[s,k,t]/sqrt(sum(T_s_k_t.^2)) for t in 1:T, r in 1:R) + sum((I_tot_r[r]+I_tot_k[k])*U_s_k[s,k]/sqrt(sum(U_s_k.^2)) for r in 1:R) for s in 1:S, k in 1:K)) #without normalize
@NLexpression(CCUS, InherSafeIndic, sum( sum((I_tot_t[t]+I_tot_r[r]+I_tot_k[k])*T_s_k_t[s,k,t]/ISISqrtExpre1 for t in 1:T, r in 1:R) + sum((I_tot_r[r]+I_tot_k[k])*U_s_k[s,k]/ISISqrtExpre2 for r in 1:R) for s in 1:S, k in 1:K) ) #without normalize
println("I am in 280")
#costs:
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
@expression(CCUS,TreatEmiss[t in 1:T],sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for s in 1:T,k in 1:K))
@expression(CCUS, TranspEmiss,TransportPow*epsilon_p)
@expression(CCUS, SinkEmiss, sum(F_k[k]*Mu_k[k] for k in 1:K))
@expression(CCUS,TotEmiss,SourceEmiss+sum(TreatEmiss)+TranspEmiss+SinkEmiss)#Emission objective
# @NLobjective(CCUS,Min,TAC)
# ###optimize total cost
# JuMP.optimize!(CCUS)
# println("TAC value:", value(TAC))
# println("Emission value:", value(TotEmiss))
# println("TAC value:", value(InherSafeIndic))
##optimize total emission
@objective(CCUS,Min,TotEmiss)
JuMP.optimize!(CCUS)
println("TAC value:", value(TAC))
println("Emission value:", value(TotEmiss))
println("TAC value:", value(InherSafeIndic))
@NLobjective(CCUS,Min,InherSafeIndic)
JuMP.optimize!(CCUS)
println("TAC value:", value(TAC))
println("Emission value:", value(TotEmiss))
println("TAC value:", value(InherSafeIndic))