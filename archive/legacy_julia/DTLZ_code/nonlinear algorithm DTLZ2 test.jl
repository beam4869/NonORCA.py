using JuMP, CPLEX, CSV, DataFrames, StatsBase, Leiden, LinearAlgebra, Ipopt, ForwardDiff, NLPModels, NLPModelsJuMP, Plots
M = 4
k = 5
n = M+k-1
DimNum = n
itera = 5
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
function NLPorcadot(Ae,ce,de,comm,NoOfSP)# Ae is the constrain matrix and ce is the objective matrix
  NoOfObj = Int(size(ce,1)/NoOfSP)
  println("NoOfObj = ", NoOfObj)
  normc=zeros(size(ce)) # normalized Cmat matrix
  posmin=1
  for i in 1:size(ce,1)
    normc[i,:]=-normalize(ce[i,:])#normalize the value of the matrix row by row
  end # thus normc is the normalized ce
  #println(normc)
  dpm=zeros(size(ce,1),size(ce,1)) # might be different if ce is not a square. Thus, make a matrix of 0 values with same # of rows and columns of the # of ce's rows
  for i in 1:size(ce,1)
    for j in i+1:size(ce,1) # the # of columes at least should be larger than the # of rows.
      dpm[i,j]=dot(normc[i,:],normc[j,:]) # the upper triangle is filled with the dots products of rows in normc, of which the length of the vectors is 1. 
    end # all the dot product results of different rows in normc are stored in dmp.
  end
  dsm=zeros(size(ce,1),size(ce,1),size(Ae,1))
  totineqwt=zeros(size(ce,1),size(ce,1),size(Ae,1))
  deltaine=zeros(size(ce,1),size(ce,1),size(Ae,1))
  #Inequality constraints
  for i in 1:size(ce,1)
    for j in i+1:size(ce,1)
      for k in 1:size(Ae,1)
        # println("principle one: ", dot(normc[i,:],Ae[k,:]))
        # println("principle two: ", dot(normc[j,:],Ae[k,:]))
    if dot(normc[i,:],Ae[k,:])!=0 && dot(normc[j,:],Ae[k,:])!=0 # if this two objective vectors are not normal to a row in the constrain matrix
        cp1=normc[i,:]-dot(normc[i,:],Ae[k,:])/(sqrt(sum(Ae[k,p]^2 for p in 1:size(Ae,2)))^2)*Ae[k,:] # to project normc[i] on Ae[K]
        cn1=normc[i,:]-cp1 # get the left part that is normal to Ae[k]
        cp2=normc[j,:]-dot(normc[j,:],Ae[k,:])/(sqrt(sum(Ae[k,p]^2 for p in 1:size(Ae,2)))^2)*Ae[k,:]
        cn2=normc[j,:]-cp2
        if cp1!=zeros(length(cp1)) 
            cp1norm=normalize(cp1) # normalize the proejcted vectors
        else
            cp1norm=cp1 # mean if cp1 is not all zero?, which means that normc[i] is acutually normal to Ae[k]
        end
        if cp2!=zeros(length(cp2))
            cp2norm=normalize(cp2)
        else
            cp2norm=cp2
        end
        #Previously we don't count the objecives that are pointing into the contraints because we can only have optimal solutions in the boundary of the feasible region in linear problem.
        #But now we are considering nonlinear problem, we need to consider all the objectives. Optimal points can be in the middle of the feasible region when objectives are nonliear.
        # if dot(Ae[k,:],cn1)>0 && dot(Ae[k,:],cn2)>0 # normally the dot product results should be zero because the cni is normal to Ae[k]
        secstrength=dot(cp1norm,cp2norm)
        deltaine[i,j,k]=secstrength#abs(secstrength-dpm[i,j])  #storage of Sij
        dsm[i,j,k]+=secstrength*(1-posmin*(1/(1+exp(-10*secstrength)))) # calculat the weight Sij*Wijk
        if secstrength==0
            totineqwt[i,j,k]=0 # why
        else
            totineqwt[i,j,k]+=(1-posmin*(1/(1+exp(-10*secstrength)))) # store the Wijk
        end
        # else
        #   # println("one here")siz
        #     secstrength=0
        #     dsm[i,j,k]+= secstrength
        #     totineqwt[i,j,k]+=0
        # end
    else#if one objective is perpendicular to the constrain, projected objective will be zero, making the Sij also zero
         secstrength=0
        dsm[i,j,k]+= secstrength
        totineqwt[i,j,k]+=0
    end
        #println(cp1,cp2)
        #println(cp1norm, cp2norm)

        #println(secstrength, " secondary strength")
        #dsm[i,j]+=(1-(secstrength+1)*0.5)*secstrength#secstrength

      end
    end
  end
  # println("ahead deltaine:", deltaine)
  # println("max deltaine:", maximum(deltaine))
  if maximum(deltaine) != 0
    deltaine=deltaine/maximum(deltaine) # rescale the biggest as one..
  end
  ##############################################################################
  #Equality constraints
  dsmeq=zeros(size(ce,1),size(ce,1),size(de,1))
  toteqwt=zeros(size(ce,1),size(ce,1),size(de,1))
  deltaeq=zeros(size(ce,1),size(ce,1),size(de,1))
  for i in 1:size(ce,1)
    for j in i+1:size(ce,1)
      for k in 1:size(de,1)
        cp1=normc[i,:]-dot(normc[i,:],de[k,:])/(sqrt(sum(de[k,p]^2 for p in 1:size(de,2)))^2)*de[k,:]
        cp2=normc[j,:]-dot(normc[j,:],de[k,:])/(sqrt(sum(de[k,p]^2 for p in 1:size(de,2)))^2)*de[k,:]

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
        if dot(normc[i,:],de[k,:])!=0 && dot(normc[j,:],de[k,:])!=0

            seceqstren=dot(cp1norm,cp2norm)
            deltaeq[k]=abs(seceqstren-dpm[i,j])# are you sure????? why[k] instead of [i,j,k]
            # deltaeq[i,j,k]=seceqstren# are you sure????? why[k] instead of [i,j,k]
            ##########################changed
            # println("deltaeq: ", deltaeq[i,j,k])
            # dsmeq[i,j,k]+=seceqstren*(1-posmin*(1/(1+exp(-10*seceqstren))))
            dsmeq[i,j,k]+=seceqstren*1#(1-posmin*(1/(1+exp(-10*seceqstren))))
            #########################changed
            # println("Sijk: ", seceqstren) # test; show the Sijk;No abnorm
            if seceqstren==0
                toteqwt[i,j,k]=0
            else
                toteqwt[i,j,k]+=(1-posmin*(1/(1+exp(-10*seceqstren)))) # might be a bug
            end
            # println("Wijk: ", toteqwt[i,j,k]) # No abnorm
        else
            seceqstren=0
            deltaeq[i,j,k]=0
            dsmeq[i,j,k]+=0
            toteqwt[i,j,k]+=0
        end
      end
    end
  end
  if maximum(deltaeq) != 0
    deltaeq=deltaeq/maximum(deltaeq)
  end
  #   println("deltaeq: ", deltaeq)


  totwt= zeros(NoOfObj,NoOfObj)
  totdsm=zeros(NoOfObj,NoOfObj)
  #deltaine=ones(size(ce,1),size(ce,1),size(Ae,1))
  #deltaeq=ones(size(ce,1),size(ce,1),size(de,1))
  for i in 1:NoOfObj
    for j in i+1:NoOfObj
      totwt[i,j]=totwt[j,i]+=sum(sum(totineqwt[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k]*deltaine[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(Ae,1))+sum(sum(toteqwt[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k]*deltaeq[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(de,1))# upper part of Sij
      # println("totineqwt: ", totineqwt[i,j,:])
      # println("deltaine: ", deltaine[i,j,:])
      # println("toteqwt: ",toteqwt[i,j,:])
      # println("deltaeq[i,j,:]: ",deltaeq[i,j,:])
      totdsm[i,j]=totdsm[j,i]+=sum(sum(dsmeq[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k]*deltaeq[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(de,1))+sum(sum(dsm[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k]*deltaine[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(Ae,1))
      # println("totdsm: ",totdsm[i,j])
      # println("dsmeq: ",dsmeq[i,j,:])
      # println("totwt: ",totwt[i,j])
      #if dsm[i,j] ==0 || totineqwt[i,j]==0
    #    totineqwt[i,j]=1
     # end
      #if dsmeq[i,j]==0 || toteqwtz[i,j]==0
    #    toteqwt[i,j]=1
     # end
    end
  end
  totwt= zeros(NoOfObj,NoOfObj)
  totdsm=zeros(NoOfObj,NoOfObj)
  #println(dsm, " ", totineqwt)
  #println(dsmeq," ", toteqwt)
  #dsmfin=zeros(size(ce,1),size(ce,1))
  #dsmeqfin=zeros(size(ce,1),size(ce,1))
  comb=zeros(NoOfObj,NoOfObj)
  combmin=zeros(NoOfObj,NoOfObj)
  for i in 1:NoOfObj
    for j in i+1:NoOfObj
      #dsmfin[i,j]=dsm[i,j]/totineqwt[i,j]
      #dsmeqfin[i,j]=dsmeq[i,j]/toteqwt[i,j]
      comb[i,j]=comb[j,i]=totdsm[i,j]/totwt[i,j]# why is like this
        # comb[i,j] = sum(dsm[i,j,k] for k in 1:size(Ae,1))/sum(toteqwt[i,j,k] for k in 1:size(de,1))

      combmin[i,j]=combmin[j,i]=minimum(vcat(dsmeq[i,j,:],dsm[i,j,:])) #debuging line.
    end
  end
  for i in 1:NoOfObj
      for j in i+1:NoOfObj
          if isnan(comb[i,j])
              comb[i,j]=comb[j,i]=0
          end
      end
  end

  for i in 1:NoOfObj
      for j in i+1:NoOfObj
          comb[i,j]=comb[j,i]=(1/2)*(comb[i,j]+1)# result of Sij
          combmin[i,j]=combmin[j,i]=(1/2)*(combmin[i,j]+1)

      end
  end
  for i in 1:NoOfObj
    comb[i,i] = 1
  end
  println(comb)
  groups=reduced(copy(comb),comm)
  
  #dpm, dsmfin, dsmeqfin, groups,
  return comb,  combmin, groups, dsmeq, dsm, dpm, deltaeq, deltaine,totdsm,totwt,toteqwt#, dsmfin, dsmeqfin, dpm
end   
function NLPorcadot_orignalpara(Ae,ce,de,comm,NoOfSP)# Ae is the constrain matrix and ce is the objective matrix
    NoOfObj = Int(size(ce,1)/NoOfSP)
    println("NoOfObj = ", NoOfObj)
    normc=zeros(size(ce)) # normalized Cmat matrix
    posmin=0.9
    beta = 100
    for i in 1:size(ce,1)
      normc[i,:]=-normalize(ce[i,:])#normalize the value of the matrix row by row
    end # thus normc is the normalized ce
    #println(normc)
    dpm=zeros(size(ce,1),size(ce,1)) # might be different if ce is not a square. Thus, make a matrix of 0 values with same # of rows and columns of the # of ce's rows
    for i in 1:size(ce,1)
      for j in i+1:size(ce,1) # the # of columes at least should be larger than the # of rows.
        dpm[i,j]=dot(normc[i,:],normc[j,:]) # the upper triangle is filled with the dots products of rows in normc, of which the length of the vectors is 1. 
      end # all the dot product results of different rows in normc are stored in dmp.
    end
    dsm=zeros(size(ce,1),size(ce,1),size(Ae,1))
    totineqwt=zeros(size(ce,1),size(ce,1),size(Ae,1))
    deltaine=zeros(size(ce,1),size(ce,1),size(Ae,1))
    #Inequality constraints
    for i in 1:size(ce,1)
      for j in i+1:size(ce,1)
        for k in 1:size(Ae,1)
          # println("principle one: ", dot(normc[i,:],Ae[k,:]))
          # println("principle two: ", dot(normc[j,:],Ae[k,:]))
            if dot(normc[i,:],Ae[k,:])!=0 && dot(normc[j,:],Ae[k,:])!=0 # if this two objective vectors are not normal to a row in the constrain matrix
                cp1=normc[i,:]-dot(normc[i,:],Ae[k,:])/(sqrt(sum(Ae[k,p]^2 for p in 1:size(Ae,2)))^2)*Ae[k,:] # to project normc[i] on Ae[K]
                cn1=normc[i,:]-cp1 # get the left part that is normal to Ae[k]
                cp2=normc[j,:]-dot(normc[j,:],Ae[k,:])/(sqrt(sum(Ae[k,p]^2 for p in 1:size(Ae,2)))^2)*Ae[k,:]
                cn2=normc[j,:]-cp2
                if cp1!=zeros(length(cp1)) 
                    cp1norm=normalize(cp1) # normalize the proejcted vectors
                else
                    cp1norm=cp1 # mean if cp1 is not all zero?, which means that normc[i] is acutually normal to Ae[k]
                end
                if cp2!=zeros(length(cp2))
                    cp2norm=normalize(cp2)
                else
                    cp2norm=cp2
                end
                #Previously we don't count the objecives that are pointing into the contraints because we can only have optimal solutions in the boundary of the feasible region in linear problem.
                #But now we are considering nonlinear problem, we need to consider all the objectives. Optimal points can be in the middle of the feasible region when objectives are nonliear.
                # if dot(Ae[k,:],cn1)>0 && dot(Ae[k,:],cn2)>0 # normally the dot product results should be zero because the cni is normal to Ae[k]
                secstrength=dot(cp1norm,cp2norm)
                deltaine[i,j,k]=secstrength#abs(secstrength-dpm[i,j])  #storage of Sij
                dsm[i,j,k]+=secstrength*(1-posmin*(1/(1+exp(-beta*secstrength)))) # calculat the weight Sij*Wijk
                if secstrength==0
                    totineqwt[i,j,k]=0 # why
                else
                    totineqwt[i,j,k]+=(1-posmin*(1/(1+exp(-beta*secstrength)))) # store the Wijk
                end
          # else
          #   # println("one here")siz
          #     secstrength=0
          #     dsm[i,j,k]+= secstrength
          #     totineqwt[i,j,k]+=0
          # end
            else#if one objective is perpendicular to the constrain, projected objective will be zero, making the Sij also zero
                secstrength=0
                dsm[i,j,k]+= secstrength
                totineqwt[i,j,k]+=0
            end
          #println(cp1,cp2)
          #println(cp1norm, cp2norm)
  
          #println(secstrength, " secondary strength")
          #dsm[i,j]+=(1-(secstrength+1)*0.5)*secstrength#secstrength
  
        end
      end
    end
    # println("dsm: ", dsm)
    # println("totineqwt: ", totineqwt)
    # println("ahead deltaine:", deltaine)
    # println("max deltaine:", maximum(deltaine))
    if maximum(deltaine) != 0
      deltaine=deltaine/maximum(deltaine) # rescale the biggest as one..
    end
    ##############################################################################
    #Equality constraints
    dsmeq=zeros(size(ce,1),size(ce,1),size(de,1))
    toteqwt=zeros(size(ce,1),size(ce,1),size(de,1))
    deltaeq=zeros(size(ce,1),size(ce,1),size(de,1))
    for i in 1:size(ce,1)
      for j in i+1:size(ce,1)
        for k in 1:size(de,1)
          cp1=normc[i,:]-dot(normc[i,:],de[k,:])/(sqrt(sum(de[k,p]^2 for p in 1:size(de,2)))^2)*de[k,:]
          cp2=normc[j,:]-dot(normc[j,:],de[k,:])/(sqrt(sum(de[k,p]^2 for p in 1:size(de,2)))^2)*de[k,:]
  
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
          if dot(normc[i,:],de[k,:])!=0 && dot(normc[j,:],de[k,:])!=0
  
              seceqstren=dot(cp1norm,cp2norm)
              deltaeq[k]=abs(seceqstren-dpm[i,j])# are you sure????? why[k] instead of [i,j,k]
              #########what is this delta for?????
              # deltaeq[i,j,k]=seceqstren# are you sure????? why[k] instead of [i,j,k]
              ##########################changed
              # println("deltaeq: ", deltaeq[i,j,k])
              # dsmeq[i,j,k]+=seceqstren*(1-posmin*(1/(1+exp(-10*seceqstren))))
              dsmeq[i,j,k]+=seceqstren*1#(1-posmin*(1/(1+exp(-10*seceqstren)))) 
              #actuall this dsmeq is exactly the Wijk*Sijk in equal constaints
              #########################changed
              # println("Sijk: ", seceqstren) # test; show the Sijk;No abnorm
              #weight in eqaulity constaint should be 1!!!
              toteqwt[i,j,k] = 1

            #   if seceqstren==0
            #       toteqwt[i,j,k]=0
            #   else
            #       toteqwt[i,j,k]+=(1-posmin*(1/(1+exp(-beta*seceqstren)))) # might be a bug
            #   end
              # println("Wijk: ", toteqwt[i,j,k]) # No abnorm
          else
              seceqstren=0
              deltaeq[i,j,k]=0
              dsmeq[i,j,k]+=0
              toteqwt[i,j,k]+=0
          end
        end
      end
    end
    if maximum(deltaeq) != 0
      deltaeq=deltaeq/maximum(deltaeq)
    end
    # println("deltaeq: ", deltaeq)
  
  
    totwt= zeros(NoOfObj,NoOfObj)
    totdsm=zeros(NoOfObj,NoOfObj)
    #deltaine=ones(size(ce,1),size(ce,1),size(Ae,1))
    #deltaeq=ones(size(ce,1),size(ce,1),size(de,1))
    for i in 1:NoOfObj
      for j in i+1:NoOfObj
        totwt[i,j]=totwt[j,i]+=sum(sum(totineqwt[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(Ae,1))+sum(sum(toteqwt[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(de,1))# upper part of Sij
        # println("totineqwt: ", totineqwt[i,j,:])
        # println("deltaine: ", deltaine[i,j,:])
        # println("toteqwt: ",toteqwt[i,j,:])
        # println("deltaeq[i,j,:]: ",deltaeq[i,j,:])
        totdsm[i,j]=totdsm[j,i]+=sum(sum(dsmeq[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(de,1))+sum(sum(dsm[s+(i-1)*NoOfSP,s+(j-1)*NoOfSP,k] for s in 1:NoOfSP) for k in 1:size(Ae,1))
        # println("totdsm: ",totdsm[i,j])
        # println("dsmeq: ",dsmeq[i,j,:])
        # println("totwt: ",totwt[i,j])
        #if dsm[i,j] ==0 || totineqwt[i,j]==0
      #    totineqwt[i,j]=1
       # end
        #if dsmeq[i,j]==0 || toteqwtz[i,j]==0
      #    toteqwt[i,j]=1
       # end
      end
    end
    println("totdsm: ", totdsm)
    println("totwt: ", totwt)
    #println(dsm, " ", totineqwt)
    #println(dsmeq," ", toteqwt)
    #dsmfin=zeros(size(ce,1),size(ce,1))
    #dsmeqfin=zeros(size(ce,1),size(ce,1))
    comb=zeros(NoOfObj,NoOfObj)
    combmin=zeros(NoOfObj,NoOfObj)
    for i in 1:NoOfObj
      for j in i+1:NoOfObj
        #dsmfin[i,j]=dsm[i,j]/totineqwt[i,j]
        #dsmeqfin[i,j]=dsmeq[i,j]/toteqwt[i,j]
        comb[i,j]=comb[j,i]=totdsm[i,j]/totwt[i,j]# why is like this
          # comb[i,j] = sum(dsm[i,j,k] for k in 1:size(Ae,1))/sum(toteqwt[i,j,k] for k in 1:size(de,1))
  
        combmin[i,j]=combmin[j,i]=minimum(vcat(dsmeq[i,j,:],dsm[i,j,:])) #debuging line.
      end
    end
    for i in 1:NoOfObj
        for j in i+1:NoOfObj
            if isnan(comb[i,j])
                comb[i,j]=comb[j,i]=0
            end
        end
    end
  
    for i in 1:NoOfObj
        for j in i+1:NoOfObj
            comb[i,j]=comb[j,i]=(1/2)*(comb[i,j]+1)# result of Sij
            combmin[i,j]=combmin[j,i]=(1/2)*(combmin[i,j]+1)
  
        end
    end
    for i in 1:NoOfObj
      comb[i,i] = 1
    end
    println(comb)
    groups=reduced(copy(comb),comm)
    
    #dpm, dsmfin, dsmeqfin, groups,
    return comb,  combmin, groups, dsmeq, dsm, dpm, deltaeq, deltaine,totdsm,totwt,toteqwt#, dsmfin, dsmeqfin, dpm
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
function NLPoriginal() 
    #optimization model
    original = Model(Ipopt.Optimizer)
    @variable(original, x[1:DimNum])
    #steady constraints. section 2.2
    @constraint(original, CircleCovnstraint, x[1]^2 + x[2]^2/4+x[3]^2/9 <= 1) 
    # @constraint(sample, outerapproximation[i in 1:length(selectedPoints)], sum(ForwardDiff.gradient(f, selectedPoints[i])[j]*(x[j]-selectedPoints[i][j]) for j in 1:DimNum) <= 0) #double check to see if larger or less than zero
    @expression(original, obj1, exp(-((x[1]-0.2)^2+x[2]^2)+x[3]))
    @expression(original, obj2, x[1]+x[2]+x[3])
    @expression(original, obj3, x[1]+x[2]^2+x[3]^3)
    @expression(original, obj4, x[1]^2+x[2]^2+x[3]^2)

    #objective
    @objective(original, Min, obj1)
    JuMP.optimize!(original) # depend on the package that you are using.
end
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
  return value.(x), original
end
function deltaGeneration(Ae,ce,de) # Ae unequal constraints. Ce objectives. De equal constraints. 
  normc=zeros(size(ce)) # normalized Cmat matrix
  delta  = zeros(size(ce,2))
  for i in 1:size(ce,1)
    normc[i,:]=-normalize(ce[i,:])#normalize the value of the matrix row by row
  end # thus normc is the normalized ce
  omega = rand(Float64, size(ce,1))*2
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
function vertexGeneration(m1,m2) 
    #optimization model
    A = []
    stoc_design=Model(Ipopt.Optimizer)
    #variables
    @variable(stoc_design, Flow[1:No_Unit])
    #steady constraints. section 2.2
    @constraint(stoc_design, FlowConstraint1[i in 1:No_Unit], Flow[i] >= 0) 

    @constraint(stoc_design, FlowConstraint2[i in 1:No_Unit], Flow[i] <= 100) 
    @constraint(stoc_design, ReactorFlow[i in 1:No_Reactor], sum(Flow[i] for i in 1:No_Reactor) == 100) 
    @constraint(stoc_design, SeparatorFlow[i in No_Reactor+1:No_Unit], sum(Flow[i] for i in No_Reactor+1:No_Unit) == sum((1-efficiency[i]/2)*Flow[i] for i in 1:No_Reactor)) 
    @constraint(stoc_design, minproduct, sum(efficiency[i]*Flow[i] for i in No_Reactor+1:No_Unit) >= 20) # changed from 25 to 20
    # z = piecewiselinear(stoc_design, Flow[1], Flow[2],Flow[3],Flow[4], 0:0.1:100, 0:0.1:100,0:0.1:100,0:0.1:100, (m,n,u,v) -> cost_ref[1]*((m/Flow_ref[1])^gamma)
    # +cost_ref[2]*((n/Flow_ref[2])^gamma)+cost_ref[3]*((u/Flow_ref[3])^gamma)+cost_ref[4]*((v/Flow_ref[4])^gamma)) # piecewise linear of the capital cost objective.
    #objective expressions
    @expression(stoc_design, costs, sum(cost_ref[n]*((Flow[n]/Flow_ref[n])^gamma) for n in 1:No_Unit)/theta+ sum(cost_para[j]*Flow[j] for j in 1:No_Unit))
    @expression(stoc_design, emms, sum(emission_para[i]*Flow[i] for i in 1:No_Unit))
    @expression(stoc_design, risk, sum(delta_safety[m1][i]*Flow[i] for i in 1:No_Unit))
    @expression(stoc_design, equity, sum(delta_equity[m2][i]*Flow[i] for i in 1:No_Unit)) 
    #epsilon constraints
    @constraint(stoc_design,epcone,emms<=epve) # emmision should less than a value
    @constraint(stoc_design,epconr,risk<=epvr) # water usage should also less than a value.
    #objective
    @NLobjective(stoc_design, Min, costs)
    JuMP.optimize!(stoc_design) # depend on the package that you are using.
    if termination_status(stoc_design)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
        return [0,0,0]
    end
    push!(A,value.(Flow))
    @objective(stoc_design, Min, emms)
    JuMP.optimize!(stoc_design) # depend on the package that you are using.
    if termination_status(stoc_design)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
        return [0,0,0]
    end
    push!(A,value.(Flow))
    @objective(stoc_design, Min, risk)
    JuMP.optimize!(stoc_design) # depend on the package that you are using.
    if termination_status(stoc_design)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
        return [0,0,0]
    end
    push!(A,value.(Flow))
    @objective(stoc_design, Min, equity)
    JuMP.optimize!(stoc_design) # depend on the package that you are using.
    if termination_status(stoc_design)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
        return [0,0,0]
    end
    push!(A,value.(Flow))
    return A
end
#generate the selected fixed points.

global selectedPoints = []

#obtain the vertexs by optimize each of the objectives individually.
vertex_list = []
DTLZ2=Model(Ipopt.Optimizer)
#variables
# @variable(stoc_design, Flow[i in 1:No_Unit],start = Flow0[i])
@variable(DTLZ2, x[1:n], start = 0.5)
#steady constraints. section 2.2
@constraint(DTLZ2, xConstraints1[i in 1:n], x[i] >= 0) 
@constraint(DTLZ2, xConstraints2[i in 1:n], x[i] <= 1) 
# @NLconstraint(DTLZ2, xMMol, sqrt(sum(x[i]^2 for i in 1:n)) == k) 

# @expression(stoc_design, costs, sum(cost_ref[n]*((Flow[n]/Flow_ref[n])^gamma) for n in 1:No_Unit)/theta+ sum(cost_para[j]*Flow[j] for j in 1:No_Unit))
# @NLexpression(DTLZ2, g, sum((x[i] - 0.5)^2 for i in 1:n))
@NLexpression(DTLZ2, f1, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*cos(x[3]*pi/2))
@NLexpression(DTLZ2, f2, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*sin(x[3]*pi/2))
@NLexpression(DTLZ2, f3, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*sin(x[2]*pi/2))
@NLexpression(DTLZ2, f4, (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(x[1]*pi/2))

#objective
# push!(vertex_list,[88.1773,11.8226,0.00001,59.1133])
@NLobjective(DTLZ2, Min, f1)
JuMP.optimize!(DTLZ2) # depend on the package that you are using.
if termination_status(DTLZ2)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
    return [0,0,0]
end
# println(111)
push!(vertex_list,value.(x))

@NLobjective(DTLZ2, Min, f2)
JuMP.optimize!(DTLZ2) # depend on the package that you are using.
if termination_status(DTLZ2)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
    return [0,0,0]
end
push!(vertex_list,value.(x))

@NLobjective(DTLZ2, Min, f3)
JuMP.optimize!(DTLZ2) # depend on the package that you are using.
if termination_status(DTLZ2)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
    return [0,0,0]
end
push!(vertex_list,value.(x))

@NLobjective(DTLZ2, Min, f4)
JuMP.optimize!(DTLZ2) # depend on the package that you are using.
if termination_status(DTLZ2)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
    return [0,0,0]
end
push!(vertex_list,value.(x))
# println(vertex_list)
#####################################
#####################################
#Find selected points by using our method
for a in 1:length(vertex_list)
  # global selectedPoints
  push!(selectedPoints,vertex_list[a])
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
    # @NLexpression(jmod, f1, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*cos(x[3]*pi/2))
    # @NLexpression(jmod, f2, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*sin(x[3]*pi/2))
    # @NLexpression(jmod, f3, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*sin(x[2]*pi/2))
    # @NLexpression(jmod, f4, (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(x[1]*pi/2))
    ff1(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*cos(x[3]*pi/2)
    ff2(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*sin(x[3]*pi/2)
    ff3(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*sin(x[2]*pi/2)
    ff4(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(x[1]*pi/2)
    @NLconstraint(jmod, obj1, sum(ForwardDiff.gradient(ff1, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    @NLconstraint(jmod, obj2, sum(ForwardDiff.gradient(ff2, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    @NLconstraint(jmod, obj3, sum(ForwardDiff.gradient(ff3, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
    @NLconstraint(jmod, obj4, sum(ForwardDiff.gradient(ff4, selectedPoints[s+(a-1)*(itera+1)])[j]*(x[j]-selectedPoints[s+(a-1)*(itera+1)][j]) for j in 1:DimNum) >= 0)
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
    println(Newpoint)
    selectedPoints = push!(selectedPoints,Newpoint)
  end
end
println("selectetd points: ", selectedPoints)

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
@constraint(DTLZ2_i, xConstraints1[i in 1:n], x[i] >= 0.00000000001) 
@constraint(DTLZ2_i, xConstraints2[i in 1:n], x[i] <= 1) 
nlp1i=MathOptNLPModel(DTLZ2_i)
xe1i=ones(nlp1i.meta.nvar)
Amats=Matrix(jac(nlp1i,xe1i))
# println("Amats: ", Amats)
jmod=Model(CPLEX.Optimizer)

@variable(jmod, x[i in 1:n]) # same as before
ff1(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*cos(x[3]*pi/2)
ff2(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*sin(x[3]*pi/2)
ff3(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*sin(x[2]*pi/2)
ff4(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(x[1]*pi/2)
@NLconstraint(jmod, obj1[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff1, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
@NLconstraint(jmod, obj2[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff2, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
@NLconstraint(jmod, obj3[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff3, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
@NLconstraint(jmod, obj4[s in 1:length(selectedPoints)], sum(ForwardDiff.gradient(ff4, selectedPoints[s])[j]*(x[j]-selectedPoints[s][j]) for j in 1:DimNum) >= 0)
nlp=MathOptNLPModel(jmod)
xe=ones(nlp.meta.nvar)
Cmat=Matrix(jac(nlp,xe)) #Jacobi of the opt program


# Sols=NLPorcadot_orignalpara(Amats,Cmat,Dmats,2,length(selectedPoints))
Sols=NLPCorrStrengGenerating(Amats,Cmat,Dmats,2,length(selectedPoints))
# Sols=NLPorcadot(Amats,Cmat,Dmats,2,length(selectedPoints))

Adj_matrix=Sols[1]
println("Grouping: ", string(Sols[3]))
println("adjacent matrix: ")
println(Adj_matrix)
# total_factor_1_2[n+24*m-24] = D[n,m,1,2]


# df1 = DataFrame(grouping = total_grouping,factor1_2 = factor1_2, factor1_3 = factor1_3,factor1_4 = factor1_4,factor2_3 = factor2_3,factor2_4 = factor2_4,factor3_4 = factor3_4)

# df10 = DataFrame(Column1 = total_grouping, Column2 = total_factor_1_2)
# CSV.write("nonlinear_design_community_detection_correctFunc_new.csv", df1)



