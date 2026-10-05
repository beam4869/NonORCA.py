using JuMP, CPLEX, CSV, DataFrames, StatsBase, Leiden, LinearAlgebra, Ipopt, ForwardDiff, NLPModels, NLPModelsJuMP, Plots# ReverseDiff
# for i in 1:0
#     println("something")
# end
# println("nothing")
# functionset = []

M = 4
k = 5
n = M+k-1
DimNum = n
# DTLZ2=Model(CPLEX.Optimizer)
# g(x) = sum((x[i] - 0.5)^2 for i in 1:n)
# for i in 1:M
#     f(x) = (1+g(x))
#     if i == 1
#         for j in 1:M-1
#             f(x) = f(x)*cos(x[j]*pi/2)
#         end
#     else
#         f(x) = (1+g(x))*sin(x[M+1-j]*pi/2)
#     end
#     f(x) = (1+)
#     push!(functionset,f(x))
# end
# @constraint(DTLZ2, xConstraints1[i in 1:n], x[i] >= 0) 
# @constraint(DTLZ2, xConstraints2[i in 1:n], x[i] <= 1) 
# @constraint(DTLZ2, xMMol, sqrt(sum(x[i]^2 for i in 1:n)) == k) 


##test
# DTLZ2=Model(Ipopt.Optimizer)
# #variables
# # @variable(stoc_design, Flow[i in 1:No_Unit],start = Flow0[i])
# @variable(DTLZ2, x[i in 1:n], start = 0.5)
# #steady constraints. section 2.2
# @constraint(DTLZ2, xConstraints1[i in 1:n], x[i] >= 0) 
# @constraint(DTLZ2, xConstraints2[i in 1:n], x[i] <= 1) 
# # @NLconstraint(DTLZ2, xMMol, sqrt(sum(x[i]^2 for i in 1:n)) == k) 

# # @expression(stoc_design, costs, sum(cost_ref[n]*((Flow[n]/Flow_ref[n])^gamma) for n in 1:No_Unit)/theta+ sum(cost_para[j]*Flow[j] for j in 1:No_Unit))
# # @NLexpression(DTLZ2, g, sum((x[i] - 0.5)^2 for i in 1:n))
# @NLexpression(DTLZ2, f1, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*cos(x[3]*pi/2))
# @NLexpression(DTLZ2, f2, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*sin(x[3]*pi/2))
# @NLexpression(DTLZ2, f3, (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*sin(x[2]*pi/2))
# @NLexpression(DTLZ2, f4, (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(x[1]*pi/2))

# #objective
# # push!(vertex_list,[88.1773,11.8226,0.00001,59.1133])
# @NLobjective(DTLZ2, Min, f2)
# JuMP.optimize!(DTLZ2) # depend on the package that you are using.
# if termination_status(DTLZ2)!=MOI.OPTIMAL # there are something wrong, so return [0,0,0]
#     return [0,0,0]
# end
# println("optimal results: ", value.(x))

#########################################
# efficiency = [0.8,0.95,0.25,0.4]
# sample_e=Model(CPLEX.Optimizer)
# @variable(sample_e, Flow[1:4])
# @constraint(sample_e, ReactorFlow[i in 1:2], sum(Flow[i] for i in 1:2) == 100) 
# @constraint(sample_e, SeparatorFlow[i in 4], sum(Flow[i] for i in 4) == sum((1-efficiency[i]/2)*Flow[i] for i in 1:2)) 
# nlp1e=MathOptNLPModel(sample_e)
# xe1e=ones(nlp1e.meta.nvar)
# Dmats=Matrix(jac(nlp1e,xe1e))
# println(Dmats)
#####################################
# selectedPoints = [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5]
# jmod=Model(CPLEX.Optimizer)
# @variable(jmod, x[i in 1:n]) # same as before
# ff1(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*cos(x[3]*pi/2)
# ff2(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*cos(x[2]*pi/2)*sin(x[3]*pi/2)
# ff3(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*cos(x[1]*pi/2)*sin(x[2]*pi/2)
# ff4(x) = (1+sum((x[i] - 0.5)^2 for i in 1:n))*sin(x[1]*pi/2)
# a = ForwardDiff.gradient(ff1, selectedPoints)
# println(a)
# @NLconstraint(jmod, obj1, sum(ForwardDiff.gradient(ff1, selectedPoints)[j]*(x[j]-selectedPoints[j]) for j in 1:DimNum) >= 0)
# @NLconstraint(jmod, obj2, sum(ForwardDiff.gradient(ff2, selectedPoints)[j]*(x[j]-selectedPoints[j]) for j in 1:DimNum) >= 0)
# @NLconstraint(jmod, obj3, sum(ForwardDiff.gradient(ff3, selectedPoints)[j]*(x[j]-selectedPoints[j]) for j in 1:DimNum) >= 0)
# @NLconstraint(jmod, obj4, sum(ForwardDiff.gradient(ff4, selectedPoints)[j]*(x[j]-selectedPoints[j]) for j in 1:DimNum) >= 0)
# nlp=MathOptNLPModel(jmod)
# xe=ones(nlp.meta.nvar)
# Cmat=Matrix(jac(nlp,xe))
# println("Cmat is here: ", Cmat)
###############test
original = Model(Ipopt.Optimizer)
@variable(original, x[1:n])
# @constraint(original, CircleCovnstraint, x[1]^2 + x[2]^2/4+x[3]^2/9 <= 1) 
@constraint(original, xConstraints1[i in 1:n], x[i] >= 0) 
@constraint(original, xConstraints2[i in 1:n], x[i] <= 1) 
#objective
@objective(original, Min, sum((x[i] - xn[i] - delta[i] )^2 for i in 1:DimNum))
JuMP.optimize!(original) # depend on the package that you are using.
return value.(x)

using JuMP
using NLopt

# Define the DTLZ2 problem
function dtlz2(n::Int, m::Int)
    model = Model(optimizer_with_attributes(NLopt.Optimizer, "algorithm" => :LD_MMA))

    # Decision variables
    @variable(model, 0 <= x[i=1:n] <= 1)

    # Objective functions
    function g(x)
        return sum((x[i] - 0.5)^2 for i in 2:n)
    end

    for i in 1:m
        @NLobjective(model, Min, (1 + g(x)) * prod(cos(0.5 * π * x[j]) for j in 1:(m - i)) * sin(0.5 * π * x[m - i + 1]))
    end

    # Solve the model
    optimize!(model)

    # Extract the solutions
    x_opt = value.(x)
    f_opt = [objective_value(model) for i in 1:m]

    return x_opt, f_opt
end
