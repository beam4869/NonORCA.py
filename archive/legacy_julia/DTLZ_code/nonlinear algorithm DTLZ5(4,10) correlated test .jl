using JuMP, CPLEX, CSV, DataFrames, StatsBase, Leiden, LinearAlgebra, Ipopt, ForwardDiff, NLPModels, NLPModelsJuMP
#= DESCRIPTION OF DTLZ5(I,M). M DENOTES THE NUMBER OF OBJECTIVES AND I DENOTES THE DIMENSIONALITY OF THE POF. 
ALL OBJECTIVES ARE TO BE MINIMIZED.
=#
M = 10
k = 10 # DLTZ 1 normally use k = 5; most of the other DTLZ tests are using 10
alpha = 1 # a parameter in objective function. 1 and 100 are used in literature.
I = 5
n = M+k-1
DimNum = n
itera = 50
g(x) = sum((x[i] - 0.5)^2 for i in 1:n)
theta = []

for i in 1:I-1
    theta_inter(x) = pi*x[i]/2
    # println(theta_inter)
    push!(theta, theta_inter)
end
for i in I:M-1
    theta_inter(x) = pi*(1+2*g(x)*x[i])/(4*(1+g(x)))
    push!(theta, theta_inter)
end

ff1(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*cos(theta[4](x))*cos(theta[5](x))*cos(theta[6](x))*cos(theta[7](x))*cos(theta[8](x))*cos(theta[9](x))
ff2(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*cos(theta[4](x))*cos(theta[5](x))*cos(theta[6](x))*cos(theta[7](x))*cos(theta[8](x))*sin(theta[9](x))
ff3(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*cos(theta[4](x))*cos(theta[5](x))*cos(theta[6](x))*cos(theta[7](x))*sin(theta[8](x))
ff4(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*cos(theta[4](x))*cos(theta[5](x))*cos(theta[6](x))*sin(theta[7](x))
ff5(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*cos(theta[4](x))*cos(theta[5](x))*sin(theta[6](x))
ff6(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*cos(theta[4](x))*sin(theta[5](x))
ff7(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*cos(theta[3](x))*sin(theta[4](x))
ff8(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*cos(theta[2](x))*sin(theta[3](x))
ff9(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(theta[1](x))*sin(theta[2](x))
ff10(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(theta[1](x))
objfuncs = [ff1,ff2,ff3,ff4,ff5,ff6,ff7,ff8,ff9,ff10]
for i in 1:M
  println(objfuncs[i](ones(n)))
end
# println(theta1)
########## model construction
DTLZ5=Model(Ipopt.Optimizer)
#variables
@variable(DTLZ5, x[1:n], start = 0.5)

@constraint(DTLZ5, xConstraints1[i in 1:n], x[i] >= 0) 
@constraint(DTLZ5, xConstraints2[i in 1:n], x[i] <= 1) 

# @NLexpression(DTLZ5, theta1[i in 1:M-1], i < I ? x[i]*pi/2 : pi*(1 + 2*gx*x[i]) / (4*(1 + gx))) #ternary conditional operator in Julia
# @NLexpression(DTLZ5, thetaexp[i in 1:M-1],theta[i](x))
@NLexpression(DTLZ5, gx,sum((x[i] - 0.5)^2 for i in 1:n))

@NLexpression(DTLZ5, f1, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*cos(pi*(1+2*gx*x[7])/(4*(1+gx)))*cos(pi*(1+2*gx*x[8])/(4*(1+gx)))*cos(pi*(1+2*gx*x[9])/(4*(1+gx))))
@NLexpression(DTLZ5, f2, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*cos(pi*(1+2*gx*x[7])/(4*(1+gx)))*cos(pi*(1+2*gx*x[8])/(4*(1+gx)))*sin(pi*(1+2*gx*x[9])/(4*(1+gx))))
@NLexpression(DTLZ5, f3, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*cos(pi*(1+2*gx*x[7])/(4*(1+gx)))*sin(pi*(1+2*gx*x[8])/(4*(1+gx))))
@NLexpression(DTLZ5, f4, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*sin(pi*(1+2*gx*x[7])/(4*(1+gx))))
@NLexpression(DTLZ5, f5, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*sin(pi*(1+2*gx*x[6])/(4*(1+gx))))
@NLexpression(DTLZ5, f6, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*sin(pi*(1+2*gx*x[5])/(4*(1+gx))))
@NLexpression(DTLZ5, f7, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*sin(pi*(1+2*gx*x[4])/(4*(1+gx))))
@NLexpression(DTLZ5, f8, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*sin(pi*x[3]/2))
@NLexpression(DTLZ5, f9, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*sin(pi*x[2]/2))
@NLexpression(DTLZ5, f10, (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(pi*x[1]/2))
@NLexpression(DTLZ5, obj[i in 1:M], [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10][i])
# global Grouping = Matrix{String}(undef, 16, 16)
global Adj_matrix = zeros(4,4)

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
#generate the selected fixed points.

global selectedPoints = []

#obtain the vertexs by optimize each of the objectives individually.
vertex_list = []
# push!(vertex_list,[1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
#objective
# push!(vertex_list,[88.1773,11.8226,0.00001,59.1133])
#######this need to be test first before the whole calculation to make sure have a reasonable results.
for i in 1:M
  @NLobjective(DTLZ5, Min, obj[i])
  JuMP.optimize!(DTLZ5) # depend on the package that you are using.
  # if termination_status(DTLZ5)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
  #     return [0,0,0]
  # end
  push!(vertex_list,value.(x))
end

# println(vertex_list)

#####################################
#####################################
#Find selected points by using our method
for a in 1:length(vertex_list)
  # global selectedPoints
  push!(selectedPoints,vertex_list[a])
  # println("what is it?",selectedPoints)
  # global f(x) f1(x) f2(x) f3(x) f4(x)
  for s in 1:itera
    global selectedPoints
    sample_i=Model(CPLEX.Optimizer)
    @variable(sample_i, x[i in 1:n]) # same as before
    #@constraint(sample_i, outerapproximation, sum(ForwardDiff.gradient(f, selectedPoints[s+(n-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(n-1)*(itera+1)][j]) for j in 1:DimNum) <= 0) #
    @constraint(sample_i, xConstraints1[i in 1:n], x[i] >= 0) 
    @constraint(sample_i, xConstraints2[i in 1:n], x[i] <= 1) 
    nlp1i=MathOptNLPModel(sample_i)
    xe1i=ones(nlp1i.meta.nvar)
    Amats=Matrix(jac(nlp1i,xe1i))

    Dmats=zeros(1,n)
    #objectives space
    
    jmod=Model(CPLEX.Optimizer)
    @variable(jmod, x[i in 1:n]) # same as before
    # println("what is selectedPoints now?", selectedPoints)
    @NLconstraint(jmod, objconst[i in 1:M], sum(ForwardDiff.gradient(objfuncs[i], selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    # @NLconstraint(jmod, obj1, sum(ForwardDiff.gradient(ff1, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    # @NLconstraint(jmod, obj2, sum(ForwardDiff.gradient(ff2, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    # @NLconstraint(jmod, obj3, sum(ForwardDiff.gradient(ff3, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    # @NLconstraint(jmod, obj4, sum(ForwardDiff.gradient(ff4, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    # @NLconstraint(jmod, obj5, sum(ForwardDiff.gradient(ff5, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    nlp=MathOptNLPModel(jmod)
    xe=ones(nlp.meta.nvar)
    Cmat=Matrix(jac(nlp,xe))

    delta = zeros(size(Cmat,2))
    delta = deltaGeneration(Amats,Cmat,Dmats)
    # println("delta: ", delta)
    Newpoint = zeros(DimNum)
    println("delta: ", delta)
    println("sp: ", selectedPoints[s+(a-1)*(itera+1)])
    Newpoint = selectedPointGeneration(delta,selectedPoints[s+(a-1)*(itera+1)])
    # println(Newpoint)
    selectedPoints = push!(selectedPoints,Newpoint)
  end
end
# println("selectetd points: ", selectedPoints)

#equality constraints
DTLZ2_e=Model(CPLEX.Optimizer)

@variable(DTLZ2_e, x[1:DimNum]) # same as before
@constraint(DTLZ2_e, zEqualConstraint, x[1]  == x[1]) 
nlp1=MathOptNLPModel(DTLZ2_e)#what does this mean
xe1=ones(nlp1.meta.nvar)
Dmats=Matrix(jac(nlp1,xe1))
println(Dmats)
#variables
# println("Dmats: ", Dmats)
#inequality constraints
DTLZ2_i=Model(CPLEX.Optimizer)
@variable(DTLZ2_i, x[i in 1:n]) # same as before
@constraint(DTLZ2_i, xConstraints1[i in 1:n], x[i] >= 0) 
@constraint(DTLZ2_i, xConstraints2[i in 1:n], x[i] <= 1) 
nlp1i=MathOptNLPModel(DTLZ2_i)
xe1i=ones(nlp1i.meta.nvar)
Amats=Matrix(jac(nlp1i,xe1i))
# println("Amats: ", Amats)
jmod=Model(CPLEX.Optimizer)

@variable(jmod, x[i in 1:n]) # same as before
@NLconstraint(jmod, objconstr[i in 1:M, s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(objfuncs[i], selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
# @NLconstraint(jmod, obj2[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff2, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
# @NLconstraint(jmod, obj3[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff3, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
# @NLconstraint(jmod, obj4[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff4, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
nlp=MathOptNLPModel(jmod)
xe=ones(nlp.meta.nvar)
Cmat=Matrix(jac(nlp,xe)) #Jacobi of the opt program


# Sols=NLPorcadot_orignalpara(Amats,Cmat,Dmats,2,length(selectedPoints))
Sols=NLPCorrStrengGenerating(Amats,Cmat,Dmats,M-I+1,length(selectedPoints))
# Sols=NLPorcadot(Amats,Cmat,Dmats,2,length(selectedPoints))

Adj_matrix=Sols[1]
println("Grouping: ", string(Sols[3]))
println("adjacent matrix: ")
println(Adj_matrix)
# total_factor_1_2[n+24*m-24] = D[n,m,1,2]


# df1 = DataFrame(grouping = total_grouping,factor1_2 = factor1_2, factor1_3 = factor1_3,factor1_4 = factor1_4,factor2_3 = factor2_3,factor2_4 = factor2_4,factor3_4 = factor3_4)

# df10 = DataFrame(Column1 = total_grouping, Column2 = total_factor_1_2)
# CSV.write("nonlinear_design_community_detection_correctFunc_new.csv", df1)



