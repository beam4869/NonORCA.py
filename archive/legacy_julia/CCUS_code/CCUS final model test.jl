#construct a sample CCUS supply chain model refering the paper Safety-driven design 
#of carbon capture utilization and storage (CCUS) supply chains: A multi-objective optimization approach
#finite difference 
using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX,LinearAlgebra,Ipopt, BARON,Leiden
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

  CCUSreg = Model(Ipopt.Optimizer)
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
  @variable(CCUSreg, x[1:Ntot] >= 0)

  # Convenience “views” (reshaped slices of x) for readability
  R  = reshape(x[rR],  S)                 # R_s[s]
  T_ = reshape(x[rT],  S, K, T)           # T_s_k_t[s,k,t]
  U_ = reshape(x[rU],  S, K)              # U_s_k[s,k]
  V_ = reshape(x[rV],  S, K, T)           # V_s_k_t[s,k,t]
  Yv = reshape(x[rYv], S, T)              # y_s_v[s,t]
  F  = reshape(x[rF],  K)                 # F_k[k]
  for s in 1:S
      set_lower_bound(R[s], L_s[s])
      set_upper_bound(R[s], M_s[s])
  end

  # ---------------------------
  # Constraints
  # ---------------------------

  # 1) Total mass balance at sources
  @constraint(CCUSreg, [s=1:S],
      R[s] == sum(T_[s,k,t] for k in 1:K, t in 1:T) +
              sum(U_[s,k]   for k in 1:K) +
              sum(V_[s,k,t] for k in 1:K, t in 1:T)
  )

  # 2) Carbon balances
  @constraint(CCUSreg, [s=1:S, k=1:K, t=1:T],
      epsilon_t[t] * (T_[s,k,t] + V_[s,k,t]) * y_s[s] == T_[s,k,t] * y_s_t[s]
  )

  @constraint(CCUSreg, [s=1:S],
      R[s]*y_s[s] ==
          sum(T_[s,k,t]*y_s_t[s] for k in 1:K, t in 1:T) +
          sum(U_[s,k]*y_s[s]     for k in 1:K) +
          sum(V_[s,k,t]*Yv[s,t]  for k in 1:K, t in 1:T)
  )

  # 3) Sink total flow and concentration requirement
  @constraint(CCUSreg, [k=1:K],
      F[k] == sum(T_[s,k,t] for s in 1:S, t in 1:T) + sum(U_[s,k] for s in 1:S)
  )

  @constraint(CCUSreg, [k=1:K],
      F[k]*Zmin_k[k] <=
          sum(T_[s,k,t]*y_s_t[s] for s in 1:S, t in 1:T) +
          sum(U_[s,k]*y_s[s]     for s in 1:S)
  )

  @constraint(CCUSreg, [k=1:K], F[k] <= Gmax_k[k])
  @constraint(CCUSreg, F[1] >= 100.0)

  @objective(CCUSreg, Min, sum((x[i] - xn[i] - delta[i] )^2 for i in 1:Ntot))
  JuMP.optimize!(CCUSreg) # depend on the package that you are using.
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
@NLexpression(CCUS, InherSafeIndic, sum(I_tot_k[k]*(sum(T_s_k_t[s,k,t] for s=1:S,t=1:T)+sum(U_s_k[s,k] for s=1:S))/Gmax_k[k] for k=1:K)+sum(I_tot_r[r]*sum(R_s[s] for s=1:S)/sum(M_s[s] for s=1:S) for r=1:R) + sum(I_tot_t[t]*sum(T_s_k_t[s,k,t]+V_s_k_t[s,k,t] for s=1:S,k=1:K)/sum(M_s[s] for s=1:S) for t=1:T))

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
@expression(CCUS,TreatEmiss[t in 1:T],sum(T_s_k_t[s,k,t]*y_s_t[s,t]*gamma_t[t] for s in 1:T,k in 1:K))
@expression(CCUS, TranspEmiss,TransportPow*epsilon_p)
@expression(CCUS, SinkEmiss, sum(F_k[k]*Mu_k[k] for k in 1:K))
@expression(CCUS,TotEmiss,SourceEmiss+sum(TreatEmiss)+TranspEmiss+SinkEmiss)#Emission objective


println("I am in 280")

##################ISI objective

# @NLobjective(CCUS,Min,InherSafeIndic)
# @NLobjective(CCUS,Min,TAC)
@NLobjective(CCUS,Min,TotEmiss)

# @NLobjective(CCUS,Max,NetCapture)
JuMP.optimize!(CCUS)

println("ISI value:", value(InherSafeIndic))
println("Net Capture value", value(NetCapture))
println("F CO2 value: ", value.(F_CO2_k))

############################
# Nonlinear grouping driver #
############################

using ForwardDiff
using NLPModels, NLPModelsJuMP
using JuMP
using CPLEX
using StatsBase
using DataFrames
using CSV
using LinearAlgebra

# --- Choose which objectives to analyze (minimization) ---
# We negate NetCapture so all objectives are “smaller is better”.
const EPS_POW = 1e-10
const USE_OBJECTIVES = (:TotEmiss, :TAC, :InherSafeIndic)

# --- Integer set sizes (never reuse decision vector names) ---
const S_ = S               # sources count (Int)
const K_ = K               # sinks count  (Int)
const T_ = T               # time count   (Int)
const R_cnt = length(I_tot_r)    # risk index count (Int)
const T_cnt = T_                 # alias for clarity

# -------- Variable pack/unpack (rename to avoid R range collisions) --------
function pack_vars(Rvar, Tvars, Uvars, Vvars, yv, Fk)
    x = Float64[]
    append!(x, vec(Rvar))                    # length S_
    append!(x, vec(Tvars))                   # length S_*K_*T_
    append!(x, vec(Uvars))                   # length S_*K_
    append!(x, vec(Vvars))                   # length S_*K_*T_
    append!(x, vec(yv))                      # length S_*T_
    append!(x, vec(Fk))                      # length K_
    return x
end

function unpack_vars(x::AbstractVector{<:Real})
    idx = 1
    Rvar  = reshape(x[idx:idx+S_-1], S_);                          idx += S_
    Tvars = reshape(x[idx:idx+S_*K_*T_-1], S_, K_, T_);            idx += S_*K_*T_
    Uvars = reshape(x[idx:idx+S_*K_-1], S_, K_);                   idx += S_*K_
    Vvars = reshape(x[idx:idx+S_*K_*T_-1], S_, K_, T_);            idx += S_*K_*T_
    yv    = reshape(x[idx:idx+S_*T_-1], S_, T_);                   idx += S_*T_
    Fk    = reshape(x[idx:idx+K_-1], K_)
    return Rvar, Tvars, Uvars, Vvars, yv, Fk
end

# -------- Objective functions (pure Julia mirrors of your JuMP algebra) --------

# Total emissions
function obj_TotEmiss(x)
    Rvar, Tvars, Uvars, Vvars, yv, Fk = unpack_vars(x)

    # Transport power
    CompressionPow = 1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S_, k in 1:K_, t in 1:T_)
    PumpPow        = 1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S_, k in 1:K_, t in 1:T_)
    TransportPow   = CompressionPow + PumpPow

    # Emission terms (match your JuMP expressions)
    SourceEmiss = sum((1 - 0.99) * Rvar[s] for s in 1:S_) # theta_s = 0.99
    TreatEmiss  = sum(Tvars[s,k,t] * y_s_t[s,t] * gamma_t[t] for s in 1:S_, k in 1:K_, t in 1:T_)
    TranspEmiss = TransportPow * epsilon_p
    SinkEmiss   = sum(Fk[k] * Mu_k[k] for k in 1:K_)

    return SourceEmiss + TreatEmiss + TranspEmiss + SinkEmiss
end


# function obj_TAC(x)
#     Rvar, Tvars, Uvars, Vvars, yv, Fk = unpack_vars(x)

#     # Transport power
#     CompressionPow = 1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
#                                 for s in 1:S_, k in 1:K_, t in 1:T_)
#     PumpPow        = 1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
#                                 for s in 1:S_, k in 1:K_, t in 1:T_)
#     TransportPow   = CompressionPow + PumpPow

#     # Treat cost  (ensure TranspCost_s_t is S×T, e.g., reshape([0,29,43,35], S, T))
#     TreatCost = sum(Tvars[s,k,t] * y_s_t[s,t] * TranspCost_s_t[s,t]
#                     for s in 1:S_, k in 1:K_, t in 1:T_)

#     # Compressor costs (cap + oper)
#     CapCompressorCost_vec = [158902 * ((sum(ComprePowPara_s_k[s,k]*(Tvars[s,k,t]+Uvars[s,k]) for t in 1:T_) / 224)^0.84) * CRF
#                              for s in 1:S_, k in 1:K_]
#     OperCompressorCost_vec = [24 * sum(ComprePowPara_s_k[s,k]*(Tvars[s,k,t]+Uvars[s,k]) for t in 1:T_) * powerprice
#                               for s in 1:S_, k in 1:K_]
#     CapCompressorCost  = reshape(CapCompressorCost_vec, S_, K_)
#     OperCompressorCost = reshape(OperCompressorCost_vec, S_, K_)
#     CompressorCost = sum(CapCompressorCost) + sum(OperCompressorCost)

#     # Pipe diameter — make Dconst and D_s_k real matrices
#     Dconst_vec = [sqrt(velocity[s,k] * Mass_s[s] *
#                        ((SinkPress_k[k] - SourcePress_s[s]) + PressDropPara_s_k[s,k]))
#                   for s in 1:S_, k in 1:K_]
#     Dconst = reshape(Dconst_vec, S_, K_)

#     # For each k, total flow from all sources (across time)
#     FlowSum_k = [sum(Tvars[s1,k,t1] for s1 in 1:S_, t1 in 1:T_) +
#                  sum(Uvars[s1,k] for s1 in 1:S_) for k in 1:K_]

#     D_s_k_vec = [((4/π) * 8.314 * temp_s[s] * FlowSum_k[k] / Dconst[s,k])^0.5
#                  for s in 1:S_, k in 1:K_]
#     D_s_k = reshape(D_s_k_vec, S_, K_)

#     PipeCost_vec = [(95230 * D_s_k[s,k] + 96904) * CRF for s in 1:S_, k in 1:K_]
#     PipeCost = reshape(PipeCost_vec, S_, K_)

#     TranspCost = sum(DistSourSink_s_k[s,k] * PipeCost[s,k] for s in 1:S_, k in 1:K_)

#     SinkCost = sum(Fk[k] * CR_sink_k[k] for k in 1:K_)

#     return TreatCost + CompressorCost + TranspCost + SinkCost
# end
function obj_TAC(x)
    Rvar, Tvars, Uvars, Vvars, yv, Fk = unpack_vars(x)

    # Transport power
    CompressionPow = 1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S_, k in 1:K_, t in 1:T_)
    PumpPow        = 1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S_, k in 1:K_, t in 1:T_)
    TransportPow   = CompressionPow + PumpPow

    # Treat cost  (TranspCost_s_t must be S×T)
    TreatCost = sum(Tvars[s,k,t] * y_s_t[s,t] * TranspCost_s_t[s,t]
                    for s in 1:S_, k in 1:K_, t in 1:T_)

    # Compressor costs (cap + oper) — guard fractional power
    cap_base = [(sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k]) for t in 1:T_) / 224) for s in 1:S_, k in 1:K_]
    CapCompressorCost_vec  = [158902 * (max(b, 0.0) + EPS_POW)^0.84 * CRF for b in cap_base]
    OperCompressorCost_vec = [24 * sum(ComprePowPara_s_k[s,k]*(Tvars[s,k,t]+Uvars[s,k]) for t in 1:T_) * powerprice
                              for s in 1:S_, k in 1:K_]
    CapCompressorCost  = reshape(CapCompressorCost_vec,  S_, K_)
    OperCompressorCost = reshape(OperCompressorCost_vec, S_, K_)
    CompressorCost = sum(CapCompressorCost) + sum(OperCompressorCost)

    # Pipe diameter — guard the sqrt
    Dconst_vec = [sqrt(velocity[s,k] * Mass_s[s] *
                       ((SinkPress_k[k] - SourcePress_s[s]) + PressDropPara_s_k[s,k]))
                  for s in 1:S_, k in 1:K_]
    Dconst = reshape(Dconst_vec, S_, K_)

    # total flow to each sink
    FlowSum_k = [sum(Tvars[s1,k,t1] for s1 in 1:S_, t1 in 1:T_) +
                 sum(Uvars[s1,k] for s1 in 1:S_) for k in 1:K_]

    D_s_k_vec = [sqrt(max(((4 / Base.pi) * 8.314 * temp_s[s] * FlowSum_k[k] / Dconst[s,k]), 0.0) + EPS_POW)
                 for s in 1:S_, k in 1:K_]
    D_s_k = reshape(D_s_k_vec, S_, K_)

    PipeCost_vec = [(95230 * D_s_k[s,k] + 96904) * CRF for s in 1:S_, k in 1:K_]
    PipeCost = reshape(PipeCost_vec, S_, K_)

    TranspCost = sum(DistSourSink_s_k[s,k] * PipeCost[s,k] for s in 1:S_, k in 1:K_)

    SinkCost = sum(Fk[k] * CR_sink_k[k] for k in 1:K_)

    return TreatCost + CompressorCost + TranspCost + SinkCost
end


# Inherent safety indicator
function obj_InherSafeIndic(x)
    Rvar, Tvars, Uvars, Vvars, yv, Fk = unpack_vars(x)

    term_k = sum(
        I_tot_k[k] * (sum(Tvars[s,k,t] for s in 1:S_, t in 1:T_) + sum(Uvars[s,k] for s in 1:S_)) / Gmax_k[k]
        for k in 1:K_
    )

    term_r = sum(
        I_tot_r[r] * (sum(Rvar[s] for s in 1:S_) / sum(M_s[s] for s in 1:S_))
        for r in 1:R_cnt
    )

    term_t = sum(
        I_tot_t[t] * (sum(Tvars[s,k,t] + Vvars[s,k,t] for s in 1:S_, k in 1:K_) / sum(M_s[s] for s in 1:S_))
        for t in 1:T_cnt
    )

    return term_k + term_r + term_t
end

# Negative NetCapture (so we minimize)
function obj_NegNetCapture(x)
    Rvar, Tvars, Uvars, Vvars, yv, Fk = unpack_vars(x)

    CompressionPow = 1000 * sum(ComprePowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S_, k in 1:K_, t in 1:T_)
    PumpPow        = 1000 * 0.8 * sum(PumpPowPara_s_k[s,k] * (Tvars[s,k,t] + Uvars[s,k])
                                for s in 1:S_, k in 1:K_, t in 1:T_)
    TransportPow   = CompressionPow + PumpPow

    F_CO2_k = [sum(Tvars[s,k,t] * y_s_t[s,t] for s in 1:S_, t in 1:T_) +
               sum(Uvars[s,k] * y_s[s] for s in 1:S_) for k in 1:K_]

    NetCapture = sum(F_CO2_k[k] * (1 - Mu_k[k]) for k in 1:K_) -
                 sum(sum(Tvars[s,k,t] * y_s_t[s,t] * gamma_t[t] for t in 1:T_, k in 1:K_) for s in 1:S_) -
                 (TransportPow/24) * (epsilon_p/1_000_000.0)

    return -NetCapture
end

# Collect objectives in chosen order
const OBJ_MAP = Dict(
    :TotEmiss => obj_TotEmiss,
    :TAC => obj_TAC,
    :InherSafeIndic => obj_InherSafeIndic,
)
objfuncs = [OBJ_MAP[k] for k in USE_OBJECTIVES]

# -------- Problem dimension and simple bounds (adjust ub to your data!) --------
function dim_and_bounds()
    nR = S_
    nT = S_*K_*T_
    nU = S_*K_
    nV = S_*K_*T_
    nY = S_*T_
    nF = K_
    DimNum = nR + nT + nU + nV + nY + nF
    lb = zeros(DimNum)
    ub = ones(DimNum)  # TODO: replace with realistic upper bounds if available
    return DimNum, lb, ub
end

DimNum, lb, ub = dim_and_bounds()
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

# -------- Build “vertex” seeds (one per objective). You can replace with real solves. --------
# vertex_list = Vector{Vector{Float64}}()
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

for _ in 1:length(objfuncs)
    push!(vertex_list, zeros(DimNum))   # feasible seed; replace with actual single-obj opt if you like
end

# -------- Δ-step exploration around each vertex (reuse your delta helpers) --------
selectedPoints = Vector{Vector{Float64}}()
iteralist = [100,80,60,40,20, 10,5]  # iterations per vertex (tune as you like)
deltalist = [0.05,0.04,0.03,0.02,0.01,0.005]  # step sizes (tune as you like)
itera = 20           # exploration depth
deltatimer = 0.02    # step size when no equalities (your deltaGeneration uses this)

for a in 1:length(vertex_list)
    push!(selectedPoints, vertex_list[a])

    for _ in 1:itera
        # Inequality “Jacobian proxy”: use bound box model (as in your delta script)
        sample_i = Model(CPLEX.Optimizer)
        @variable(sample_i, x[1:DimNum])
        R  = reshape(x[rR],  S)            # R_s[s]
        T_ = reshape(x[rT],  S, K, T)      # T_s_k_t[s,k,t]
        U_ = reshape(x[rU],  S, K)         # U_s_k[s,k]
        V_ = reshape(x[rV],  S, K, T)      # V_s_k_t[s,k,t]
        Yv = reshape(x[rYv], S, T)         # y_s_v[s,t]
        F  = reshape(x[rF],  K)            # F_k[k]
        @constraint(sample_i, [i=1:DimNum], x[i] >= 0)
        @constraint(sample_i, [k=1:K],
        F[k]*Zmin_k[k] <=
        sum(T_[s,k,t]*y_s_t[s] for s in 1:S, t in 1:T) +
        sum(U_[s,k]*y_s[s]     for s in 1:S)
        )
        nlp_i = MathOptNLPModel(sample_i)
        xe_i  = ones(nlp_i.meta.nvar)
        Amats = Matrix(jac(nlp_i, xe_i))     # acts like Ae
        # println("Amats: ", Amats)

        # No equalities ⇒ empty / zero row
        sample_e = Model(CPLEX.Optimizer)
        @variable(sample_e, x[1:DimNum])
        R  = reshape(x[rR],  S)            # R_s[s]
        T_ = reshape(x[rT],  S, K, T)      # T_s_k_t[s,k,t]
        U_ = reshape(x[rU],  S, K)         # U_s_k[s,k]
        V_ = reshape(x[rV],  S, K, T)      # V_s_k_t[s,k,t]
        Yv = reshape(x[rYv], S, T)         # y_s_v[s,t]
        F  = reshape(x[rF],  K)            # F_k[k]
        @constraint(sample_e, [s=1:S],
          R[s] == sum(T_[s,k,t] for k in 1:K, t in 1:T) +
            sum(U_[s,k]   for k in 1:K) +
            sum(V_[s,k,t] for k in 1:K, t in 1:T)
        )

        @constraint(sample_e, [s=1:S, k=1:K, t=1:T],
          epsilon_t[t] * (T_[s,k,t] + V_[s,k,t]) * y_s[s] == T_[s,k,t] * y_s_t[s]
        )

        @constraint(sample_e, [s=1:S],
            R[s]*y_s[s] ==
                sum(T_[s,k,t]*y_s_t[s] for k in 1:K, t in 1:T) +
                sum(U_[s,k]*y_s[s]     for k in 1:K) +
                sum(V_[s,k,t]*Yv[s,t]  for k in 1:K, t in 1:T)
        )

        # 3) Sink total flow and concentration requirement
        @constraint(sample_e, [k=1:K],
            F[k] == sum(T_[s,k,t] for s in 1:S, t in 1:T) + sum(U_[s,k] for s in 1:S)
        )
        nlp_e = MathOptNLPModel(sample_e)
        xe_e  = ones(nlp_e.meta.nvar)
        Dmats = Matrix(jac(nlp_e, xe_e))
        # println("Dmats: ", Dmats)


        # Objective Jacobian at current point (linearized feasibility)
        sp = selectedPoints[end]
        jmod = Model(CPLEX.Optimizer)
        @variable(jmod, z[1:DimNum])
        @NLconstraint(jmod, objconstr[i in 1:length(objfuncs)],
            sum(ForwardDiff.gradient(objfuncs[i], sp)[j] * (z[j] - sp[j]) for j in 1:DimNum) >= 0)

        nlp = MathOptNLPModel(jmod)
        xe  = ones(nlp.meta.nvar)
        Cmat = Matrix(jac(nlp, xe))
        # println("Cmat: ", Cmat)

        Δ = deltaGeneration(Amats, Cmat, Dmats)        # your function
        println("Delta norm: ", norm(Δ))
        println("Delta: ", Δ)
        println("At point: ", sp)
        newpt = selectedPointGeneration(Δ, sp)         # your function
        push!(selectedPoints, newpt)
    end
end

# -------- Assemble global Jacobians across all selected points and group --------
Kpart = 2  # number of groups you want from Leiden (change as needed)

jmod_Obj = Model(CPLEX.Optimizer)
@variable(jmod_Obj, z[1:DimNum])
@NLconstraint(jmod_Obj, objconstr[i in 1:length(objfuncs), s in 1:length(selectedPoints)],
    sum(ForwardDiff.gradient(objfuncs[i], selectedPoints[s])[j] * (z[j] - selectedPoints[s][j]) for j in 1:DimNum) >= 0)

nlp_obj = MathOptNLPModel(jmod_Obj)
xe_obj  = ones(nlp_obj.meta.nvar)
Cmat_all = Matrix(jac(nlp_obj, xe_obj))

# Inequality Jacobian proxy (bounds)
jmod_in = Model(CPLEX.Optimizer)
 @variable(jmod_in, x[1:DimNum])
        # R  = reshape(x[rR],  S)            # R_s[s]
        T_in = reshape(x[rT],  S, K, T)      # T_s_k_t[s,k,t]
        U_in = reshape(x[rU],  S, K)         # U_s_k[s,k]
        # V_ = reshape(x[rV],  S, K, T)      # V_s_k_t[s,k,t]
        # Yv = reshape(x[rYv], S, T)         # y_s_v[s,t]
        F_in  = reshape(x[rF],  K)            # F_k[k]
        @constraint(jmod_in, [i=1:DimNum], x[i] >= 0)
        @constraint(jmod_in, [k=1:K],
        F_in[k]*Zmin_k[k] <=
        sum(T_in[s,k,t]*y_s_t[s] for s in 1:S, t in 1:T) +
        sum(U_in[s,k]*y_s[s]     for s in 1:S)
        )
nlp_in = MathOptNLPModel(jmod_in)
xe_in = ones(nlp_bounds.meta.nvar)
Amats = Matrix(jac(nlp_in, xe_in))

jmod_e = Model(CPLEX.Optimizer)
    @variable(jmod_e, x[1:DimNum])
    R_e  = reshape(x[rR],  S)            # R_s[s]
    T_e = reshape(x[rT],  S, K, T)      # T_s_k_t[s,k,t]
    U_e = reshape(x[rU],  S, K)         # U_s_k[s,k]
    V_e = reshape(x[rV],  S, K, T)      # V_s_k_t[s,k,t]
    Yv_e = reshape(x[rYv], S, T)         # y_s_v[s,t]
    F_e  = reshape(x[rF],  K)            # F_k[k]
    @constraint(jmod_e, [s=1:S],
        R_e[s] == sum(T_e[s,k,t] for k in 1:K, t in 1:T) +
        sum(U_e[s,k]   for k in 1:K) +
        sum(V_e[s,k,t] for k in 1:K, t in 1:T)
    )

    @constraint(jmod_e, [s=1:S, k=1:K, t=1:T],
        epsilon_t[t] * (T_e[s,k,t] + V_e[s,k,t]) * y_s[s] == T_e[s,k,t] * y_s_t[s]
    )

    @constraint(jmod_e, [s=1:S],
        R_e[s]*y_s[s] ==
            sum(T_e[s,k,t]*y_s_t[s] for k in 1:K, t in 1:T) +
            sum(U_e[s,k]*y_s[s]     for k in 1:K) +
            sum(V_e[s,k,t]*Yv_e[s,t]  for k in 1:K, t in 1:T)
    )

    # 3) Sink total flow and concentration requirement
    @constraint(jmod_e, [k=1:K],
        F_e[k] == sum(T_e[s,k,t] for s in 1:S, t in 1:T) + sum(U_e[s,k] for s in 1:S)
    )
nlp_e = MathOptNLPModel(jmod_e)
xe_e  = ones(nlp_e.meta.nvar)
Dmats = Matrix(jac(nlp_e, xe_e))  # no equalities

Adj, totwt, groups, totSecWt, EqSecstrength, totEqWt, IneqSecstrength, totIneqWt =
    NLPCorrStrengGenerating(Amats, Cmat_all, Dmats, Kpart, length(selectedPoints))

println("Adjacency matrix:\n", Adj)
println("Grouping (Leiden, K = $Kpart): ", groups)

# -------- Save a concise summary --------
df = DataFrame(
    ObjectiveSet = [string(USE_OBJECTIVES)],
    NumSelectedPoints = [length(selectedPoints)],
    Partitions = [Kpart],
    Grouping = [string(groups)]
)
CSV.write("CCUS_nonlinear_grouping.csv", df)
println("Saved: CCUS_nonlinear_grouping.csv")