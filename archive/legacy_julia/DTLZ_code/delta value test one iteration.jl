using JuMP, CPLEX, CSV, DataFrames, StatsBase, Leiden, LinearAlgebra, Ipopt, ForwardDiff, NLPModels, NLPModelsJuMP, Plots
#= DESCRIPTION OF DTLZ5(I,M). M DENOTES THE NUMBER OF OBJECTIVES AND I DENOTES THE DIMENSIONALITY OF THE POF. 
ALL OBJECTIVES ARE TO BE MINIMIZED.
=#
# [1.0 0.9999582428561841 0.9999448624694545 0.9999178826545199 0.9998905822091253 0.9997461556318734 0.9996787236406215 0.9997559383867745 0.9996974953655251 0.9995663260599297 0.9993880654251194 0.9993580284171747 0.9978943194370249 0.9989272656039112 0.9598157194454646 0.8284574814857053 0.7681158790169657 0.7373851950148497 0.7378615902326369 0.9566275113694093; 0.9999582428561841 1.0 0.9999442962582571 0.9999104921939828 0.9998826022136599 0.9997222066984051 0.9996541013528737 0.9997326365313002 0.9996711503935023 0.999528673244495 0.9993403745683662 0.9993139259956625 0.997812055146627 0.9988668086505075 0.9596351153592731 0.8279859394216853 0.7675042009523291 0.7368865028262993 0.7369371478856509 0.9565014478552619; 0.9999448624694545 0.9999442962582571 1.0 0.9999298874292142 0.9999059769814905 0.999767392680894 0.9997008107425633 0.9997722758927221 0.9997151224172256 0.9995819561773114 0.9993995239177247 0.9993745484838567 0.9978978854180549 0.9989415325741577 0.9597869543962974 0.8282626437756353 0.7678066786598212 0.7371086283057667 0.7374889147162318 0.95624408147523; 0.9999178826545199 0.9999104921939828 0.9999298874292142 1.0 0.9999325525012183 0.9998413805143826 0.9997907396912165 0.9998426650736585 0.9997967357357722 0.9996911827430194 0.9995397547192706 0.9995113395439428 0.9981802623490166 0.9991302357114351 0.9604038385154308 0.8299095448759652 0.7695840240656333 0.7388344704867211 0.7390863439104582 0.9554257516953188; 0.9998905822091253 0.9998826022136599 0.9999059769814905 0.9999325525012183 1.0 0.9998669885208633 0.9998196229878986 0.999859652332507 0.9998172296671675 0.9997168326257126 0.9995665882068636 0.99954300721131 0.9982267624543899 0.999168326440228 0.9605497099009579 0.8299125461938945 0.7695155291486097 0.7386710995437342 0.7387049598361913 0.9551145606432503; 0.9997461556318734 0.9997222066984051 0.999767392680894 0.9998413805143826 0.9998669885208633 1.0 0.9998936879739703 0.9998836805814153 0.9998598557407994 0.9998069012751187 0.9997058263859245 0.999680851542559 0.9986343846390129 0.9994042539316723 0.961612357533469 0.8328476979129379 0.7725931514853883 0.7419257368858787 0.7411678895394679 0.9528667045149177; 0.9996787236406215 0.9996541013528737 0.9997008107425633 0.9997907396912165 0.9998196229878986 0.9998936879739703 1.0 0.9999130672309795 0.9999033771683844 0.9998771332272118 0.9998027878769395 0.9997871543537027 0.9988660249985715 0.9995633015773656 0.9621287211513256 0.8338224083465987 0.7734565385570943 0.7425920355139096 0.7415231192387183 0.9525676792675872; 0.9997559383867745 0.9997326365313002 0.9997722758927221 0.9998426650736585 0.999859652332507 0.9998836805814153 0.9999130672309795 1.0 0.9999345964088342 0.9998872479162213 0.999797499213737 0.9997756352648655 0.9987498810053341 0.9995145279493751 0.9615756036590808 0.8333138363077055 0.7730359280555723 0.7422938609918907 0.7413283354384091 0.952788724793632; 0.9996974953655251 0.9996711503935023 0.9997151224172256 0.9997967357357722 0.9998172296671675 0.9998598557407994 0.9999033771683844 0.9999345964088342 1.0 0.9999123964757509 0.9998345237946165 0.9998159835264946 0.9988567566234605 0.9995800503168633 0.9617889466689868 0.8337739291041717 0.773498232740389 0.7427385444734824 0.7414345079143927 0.9522965884429426; 0.9995663260599297 0.999528673244495 0.9995819561773114 0.9996911827430194 0.9997168326257126 0.9998069012751187 0.9998771332272118 0.9998872479162213 0.9999123964757509 1.0 0.999897272133635 0.9998772442956694 0.9990952974976113 0.9997097009571563 0.9627045621619168 0.8357394910432967 0.7756456389831464 0.7450040439917403 0.7434677459540701 0.9513326935945593; 0.9993880654251194 0.9993403745683662 0.9993995239177247 0.9995397547192706 0.9995665882068636 0.9997058263859245 0.9998027878769395 0.999797499213737 0.9998345237946165 0.999897272133635 1.0 0.9998996880781523 0.9993408369302559 0.9998049647462122 0.9632433076510989 0.8383132468038565 0.778343033257019 0.7476609670655463 0.7455041460054159 0.950176151549746; 0.9993580284171747 0.9993139259956625 0.9993745484838567 0.9995113395439428 0.99954300721131 0.999680851542559 0.9997871543537027 0.9997756352648655 0.9998159835264946 0.9998772442956694 0.9998996880781523 1.0 0.9993425195884491 0.9998207974149003 0.963337189725031 0.8373859709740379 0.7772191216020762 0.7462701579404281 0.7436832830134326 0.9503943155108253; 0.9978943194370249 0.997812055146627 0.9978978854180549 0.9981802623490166 0.9982267624543899 0.9986343846390129 0.9988660249985715 0.9987498810053341 0.9988567566234605 0.9990952974976113 0.9993408369302559 0.9993425195884491 1.0 0.9996968853099393 0.9659802689858114 0.8489490761674068 0.7892293678518558 0.757865418763539 0.7520643265751843 0.9452529265729674; 0.9989272656039112 0.9988668086505075 0.9989415325741577 0.9991302357114351 0.999168326440228 0.9994042539316723 0.9995633015773656 0.9995145279493751 0.9995800503168633 0.9997097009571563 0.9998049647462122 0.9998207974149003 0.9996968853099393 1.0 0.9645693154002517 0.8413372405811205 0.7813322648607017 0.750244034411804 0.7463512099542721 0.9485322360349189; 0.9598157194454646 0.9596351153592731 0.9597869543962974 0.9604038385154308 0.9605497099009579 0.961612357533469 0.9621287211513256 0.9615756036590808 0.9617889466689868 0.9627045621619168 0.9632433076510989 0.963337189725031 0.9659802689858114 0.9645693154002517 1.0 0.8706101100345903 0.8069233418176593 0.7740289998420316 0.7647737962906711 0.9080139870218238; 0.8284574814857053 0.8279859394216853 0.8282626437756353 0.8299095448759652 0.8299125461938945 0.8328476979129379 0.8338224083465987 0.8333138363077055 0.8337739291041717 0.8357394910432967 0.8383132468038565 0.8373859709740379 0.8489490761674068 0.8413372405811205 0.8706101100345903 1.0 0.9372323938720637 0.8990993900297262 0.861459884210946 0.7406463972897115; 0.7681158790169657 0.7675042009523291 0.7678066786598212 0.7695840240656333 0.7695155291486097 0.7725931514853883 0.7734565385570943 0.7730359280555723 0.773498232740389 0.7756456389831464 0.778343033257019 0.7772191216020762 0.7892293678518558 0.7813322648607017 0.8069233418176593 0.9372323938720637 1.0 0.9522993306763411 0.8984469459321147 0.6686906550169314; 0.7373851950148497 0.7368865028262993 0.7371086283057667 0.7388344704867211 0.7386710995437342 0.7419257368858787 0.7425920355139096 0.7422938609918907 0.7427385444734824 0.7450040439917403 0.7476609670655463 0.7462701579404281 0.757865418763539 0.750244034411804 0.7740289998420316 0.8990993900297262 0.9522993306763411 1.0 0.9159126884519854 0.6344277575304373; 0.7378615902326369 0.7369371478856509 0.7374889147162318 0.7390863439104582 0.7387049598361913 0.7411678895394679 0.7415231192387183 0.7413283354384091 0.7414345079143927 0.7434677459540701 0.7455041460054159 0.7436832830134326 0.7520643265751843 0.7463512099542721 0.7647737962906711 0.861459884210946 0.8984469459321147 0.9159126884519854 1.0 0.6411694628274726; 0.9566275113694093 0.9565014478552619 0.95624408147523 0.9554257516953188 0.9551145606432503 0.9528667045149177 0.9525676792675872 0.952788724793632 0.9522965884429426 0.9513326935945593 0.950176151549746 0.9503943155108253 0.9452529265729674 0.9485322360349189 0.9080139870218238 0.7406463972897115 0.6686906550169314 0.6344277575304373 0.6411694628274726 1.0]

iiconst = 1
jjconst = 1
M = 12
k = 10 # DLTZ 1 normally use k = 5; most of the other DTLZ tests are using 10
alpha = 1 # a parameter in objective function. 1 and 100 are used in literature.
deltatimer = 0.0005
I = 5
n = M+k-1
DimNum = n
results_matrix  =  Matrix{Any}(undef, iiconst*jjconst, 7)# to list the type of DTLZ(I,M),No of selected points, total No of Sp, iteration number and the grouping results
for ii in 1:iiconst
    for jj in 1:jjconst
starttime = time() # record the initial time
# itera = 20
itera = 80+ii*5
g(x) = sum((x[i] - 0.5)^2 for i in 1:n)



theta = Vector{Function}(undef, M-1)

# Define theta functions for first part
for i in 1:I-1
    theta[i] = x -> pi*x[i]/2
end

# Define theta functions for second part
for i in I:M-1
    theta[i] = x -> pi*(1+2*g(x)*x[i])/(4*(1+g(x)))
end

objfuncs = Vector{Function}(undef, M)

# Define all objfuncs functions using correct scopes
for i in 1:M
    if i == 1
        objfuncs[i] = x -> begin
            prod = (1 + alpha*g(x))
            for j in 1:M-i
                prod *= cos(theta[j](x))
            end
            prod
        end
    elseif i == M
        objfuncs[i] = x -> (1 + alpha*g(x)) * sin(theta[1](x))
    else
        objfuncs[i] = x -> begin
            prod = (1 + alpha*g(x)) * sin(theta[M-i+1](x))
            for j in 1:M-i
                prod *= cos(theta[j](x))
            end
            prod
        end
    end
end
# for i in 1:M
#   println(objfuncs[i](ones(n)))
# end


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
thetaexpr = []
for i in 1:I-1
    append!(thetaexpr, [@NLexpression(DTLZ5, pi*x[i]/2)])
end
for i in I:M-1
    append!(thetaexpr, [@NLexpression(DTLZ5, pi*(1 + 2*gx*x[i]) / (4 * (1 + gx)))])
end
# thetaexpr = [@NLexpression(DTLZ5, pi*x[i]/2) for i in 1:I-1]
# append!(thetaexpr, [@NLexpression(DTLZ5, pi*(1 + 2*gx*x[i]) / (4 * (1 + gx))) for i in I:M-1])
# @NLexpression(DTLZ5, f1, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*cos(pi*(1+2*gx*x[7])/(4*(1+gx)))*cos(pi*(1+2*gx*x[8])/(4*(1+gx)))*cos(pi*(1+2*gx*x[9])/(4*(1+gx))))
# @NLexpression(DTLZ5, f2, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*cos(pi*(1+2*gx*x[7])/(4*(1+gx)))*cos(pi*(1+2*gx*x[8])/(4*(1+gx)))*sin(pi*(1+2*gx*x[9])/(4*(1+gx))))
# @NLexpression(DTLZ5, f3, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(pi*x[1]/2)*cos(pi*x[2]/2)*cos(pi*x[3]/2)*cos(pi*(1+2*gx*x[4])/(4*(1+gx)))*cos(pi*(1+2*gx*x[5])/(4*(1+gx)))*cos(pi*(1+2*gx*x[6])/(4*(1+gx)))*cos(pi*(1+2*gx*x[7])/(4*(1+gx)))*sin(pi*(1+2*gx*x[8])/(4*(1+gx))))
obj = []
for i in 1:M
    if i == 1
        append!(obj,[@NLexpression(DTLZ5,(1+gx)*prod(cos(thetaexpr[j]) for j in 1:M-1))])
    elseif i == M  
        append!(obj,[@NLexpression(DTLZ5,(1+gx)*sin(thetaexpr[1]))])

    else
        append!(obj,[@NLexpression(DTLZ5,(1+gx)*prod(cos(thetaexpr[j]) for j in 1:M-i)*sin(thetaexpr[M-i+1]))])
    end
end

# global Grouping = Matrix{String}(undef, 16, 16)
global Adj_matrix = zeros(4,4)

function reduced(A,red)
    for i in 1:size(A,1)
        for j in 1:size(A,1)
            A[i,j]=round(A[i,j]*100000,digits=0) # make the numbers be round in the first digit
            # A[i,j]=round(A[i,j],digits=5) # make the numbers be round in the first digit
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
    delta = deltatimer*sum(sum(omega[i]*cpik[i,k,:] for i in 1:size(ce,1))/sum(omega[i] for i in 1:size(ce,1)) for k in 1:size(Ae,1))/(size(Ae,1))
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
    nlp=MathOptNLPModel(jmod)
    xe=ones(nlp.meta.nvar)
    Cmat=Matrix(jac(nlp,xe))

    delta = zeros(size(Cmat,2))
    delta = deltaGeneration(Amats,Cmat,Dmats)
    # println("delta: ", delta)
    Newpoint = zeros(DimNum)
    # println("delta: ", delta)
    # println("sp: ", selectedPoints[s+(a-1)*(itera+1)])
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
# println(Dmats)
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
nlp=MathOptNLPModel(jmod)
xe=ones(nlp.meta.nvar)
Cmat=Matrix(jac(nlp,xe)) #Jacobi of the opt program


# Sols=NLPorcadot_orignalpara(Amats,Cmat,Dmats,2,length(selectedPoints))
Sols=NLPCorrStrengGenerating(Amats,Cmat,Dmats,I,length(selectedPoints))
# Sols=NLPorcadot(Amats,Cmat,Dmats,2,length(selectedPoints))
endtime = time()
Adj_matrix=Sols[1]
# println("Grouping: ", string(Sols[3]))
# println("adjacent matrix: ")
# println(Adj_matrix)
# total_factor_1_2[n+24*m-24] = D[n,m,1,2]
NototalSP = length(selectedPoints)
results_matrix[(ii-1)*jjconst+jj, 1] = "DTLZ($I, $M)"
results_matrix[(ii-1)*jjconst+jj, 2] = "$itera"
results_matrix[(ii-1)*jjconst+jj, 3] = "$NototalSP"
results_matrix[(ii-1)*jjconst+jj, 4] = "$jj"
results_matrix[(ii-1)*jjconst+jj, 5] = string(Sols[3])
results_matrix[(ii-1)*jjconst+jj, 6] = endtime - starttime
results_matrix[(ii-1)*jjconst+jj, 7] = Adj_matrix
end 
end
println(results_matrix)
df1 = DataFrame(results_matrix, :auto)
rename!(df1, ["test data set", "Number of SP within one vertex", "total Number of SP", "iteration number", "resulting grouping", "operation time (s)", "Adjacent Matrix"] )
# # CSV.write("DTLZ($I, $M) Number of selected points test.csv", df1)
CSV.write("DTLZ($I, $M) delta size of $deltatimer one iteration.csv", df1)


# df1 = DataFrame(grouping = total_grouping,factor1_2 = factor1_2, factor1_3 = factor1_3,factor1_4 = factor1_4,factor2_3 = factor2_3,factor2_4 = factor2_4,factor3_4 = factor3_4)

# df10 = DataFrame(Column1 = total_grouping, Column2 = total_factor_1_2)
# CSV.write("nonlinear_design_community_detection_correctFunc_new.csv", df1)

