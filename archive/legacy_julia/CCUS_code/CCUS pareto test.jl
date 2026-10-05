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

function solve_once(objective, ; eps_ISI=nothing, eps_Emiss=nothing, eps_Cost=nothing, solver=:BARON)
    # Build the model
    bm = build_model(solver=solver)
    
    # Set the objective
    @objective(bm.m, Min, getfield(bm, Symbol(objective)))

    # Apply ε-constraints
    if !isnothing(eps_ISI)
        @constraint(bm.m, eps_ISI_constraint, InherSafeIndic <= eps_ISI)
    end
    if !isnothing(eps_Emiss)
        @constraint(bm.m, eps_Emiss_constraint, TotEmiss <= eps_Emiss)
    end
    if !isnothing(eps_Cost)
        @constraint(bm.m, eps_Cost_constraint, TAC <= eps_Cost)
    end

    # Solve the model
    optimize!(bm.m)

    # Extract results
    status = termination_status(bm.m)
    results = Dict{Symbol,Any}(
        :status => status,
        :TAC => value(bm.TAC),
        :Emiss => value(bm.TotEmiss),
        :ISI => value(bm.InherSafeIndic)
    )

    return results
end

############# 3D PARETO FRONTIER CALCULATION #############

"""
    calculate_and_plot_3d_frontier(;
        isi_range=(9.1, 11.6), n_isi=10,
        n_inner_points=10,
        solver=:BARON
    )

Performs a structured 3-stage epsilon-constraint method to find the 3D Pareto frontier.

1. Outer loop: Constrains ISI over `isi_range` with `n_isi` points.
2. Intermediate: For each ISI, finds the min/max range for Emissions.
3. Inner loop: For each ISI, computes a 2D Pareto frontier for TAC vs. Emissions.
4. Filters the collected points to find the 3D non-dominated set.
5. Plots the 3D frontier and a 2D projection.

Returns a DataFrame of the Pareto points.
"""
function calculate_and_plot_3d_frontier(;
    isi_range=(9.1, 11.6), n_isi=10,
    n_inner_points=10,
    solver=:BARON
)
    # Use the existing non-dominated filter function
    # Ensure it handles NaNs from infeasible solves
    function _nondominated_3d_safe(points::Vector{<:NamedTuple})
        # First, filter out any points that have NaN values
        feasible_points = filter(p -> !isnan(p.TAC) && !isnan(p.Emiss) && !isnan(p.ISI), points)
        
        if isempty(feasible_points)
            return feasible_points
        end

        keep = trues(length(feasible_points))
        @inbounds for i in eachindex(feasible_points)
            keep[i] || continue
            pi = feasible_points[i]
            for j in eachindex(feasible_points)
                (i == j || !keep[j]) && continue
                pj = feasible_points[j]
                # Dominance check
                if (pj.TAC <= pi.TAC && pj.Emiss <= pi.Emiss && pj.ISI <= pi.ISI) &&
                   (pj.TAC < pi.TAC || pj.Emiss < pi.Emiss || pj.ISI < pi.ISI)
                    keep[i] = false
                    break
                end
            end
        end
        return feasible_points[keep]
    end

    println("Starting 3D Pareto frontier calculation...")
    
    isi_grid = range(isi_range[1], isi_range[2], length=n_isi)
    all_points = NamedTuple[]
    total_solves = 0

    # 1. Outer loop over ISI constraints
    for (i, eps_isi) in enumerate(isi_grid)
        println("\n--- Outer Loop Step $i/$n_isi: Constraining ISI <= $(round(eps_isi, digits=3)) ---")

        # 2. Find the range for Emissions at this ISI level
        println("  Finding Emission range for this ISI level...")
        
        # Find minimum possible emission for this ISI
        min_emiss_solve = solve_once(:emiss; eps_ISI=eps_isi, solver=solver)
        total_solves += 1
        
        # Find the emission value when minimizing for cost for this ISI
        min_tac_solve = solve_once(:cost; eps_ISI=eps_isi, solver=solver)
        total_solves += 1

        if min_emiss_solve.status == :INFEASIBLE || min_tac_solve.status == :INFEASIBLE
            println("  Infeasible to find Emission range for ISI = $eps_isi. Skipping.")
            continue
        end

        emiss_lo = min_emiss_solve.Emiss
        emiss_hi = min_tac_solve.Emiss
        
        if isnan(emiss_lo) || isnan(emiss_hi) || emiss_lo > emiss_hi
             println("  Could not determine a valid Emission range. Lo: $emiss_lo, Hi: $emiss_hi. Skipping.")
             continue
        end

        println("  Emission range found: [$(round(emiss_lo, digits=2)), $(round(emiss_hi, digits=2))]")
        
        # 3. Inner loop: Calculate TAC vs. Emission frontier for the fixed ISI
        emiss_grid = range(emiss_lo, emiss_hi, length=n_inner_points)
        
        println("  Starting inner loop for TAC vs. Emission...")
        for (j, eps_emiss) in enumerate(emiss_grid)
            # Minimize TAC with constraints on both ISI and Emission
            res = solve_once(:cost; eps_ISI=eps_isi, eps_Emiss=eps_emiss, solver=solver)
            total_solves += 1
            
            if res.status == MOI.OPTIMAL || res.status == MOI.LOCALLY_SOLVED
                point = (TAC=res.TAC, Emiss=res.Emiss, ISI=res.ISI)
                push!(all_points, point)
                println("    Inner solve $j/$n_inner_points: Feasible point found.")
            else
                println("    Inner solve $j/$n_inner_points: Infeasible.")
            end
        end
    end

    println("\nCalculation complete. Total solves: $total_solves.")
    println("Total feasible points found before filtering: $(length(all_points))")

    if !isempty(all_points)
        println("--- All Feasible Points Found (Before Filtering) ---")
        println(DataFrame(all_points))
        println("----------------------------------------------------")
    end

    # 4. Filter for the true 3D Pareto set
    println("\nFiltering for 3D non-dominated Pareto set...")
    pareto_points = _nondominated_3d_safe(all_points)
    println("Found $(length(pareto_points)) Pareto optimal points after filtering.")

    if isempty(pareto_points)
        println("No Pareto points found. Cannot plot.")
        return DataFrame()
    end

    # 5. Plotting
    df_pareto = DataFrame(pareto_points)
    df_all = DataFrame(all_points)

    # Plot 1: 3D Scatter
    p3d = scatter(
        df_pareto.Emiss, df_pareto.ISI, df_pareto.TAC,
        xlabel="Emission", ylabel="ISI", zlabel="TAC",
        title="3D Pareto Frontier",
        legend=false,
        markersize=4, markerstrokewidth=0
    )
    display(p3d)

    # Plot 2: 2D Projection
    p2d = scatter(
        df_pareto.ISI, df_pareto.Emiss,
        marker_z=df_pareto.TAC,
        colorbar=true,
        colorbar_title="TAC",
        xlabel="ISI",
        ylabel="Emission",
        title="2D Projection of Pareto Frontier (Color by TAC)",
        legend=false,
        markersize=5
    )
    display(p2d)
    p2d_all = scatter(
        df_all.ISI, df_all.Emiss,
        marker_z=df_all.TAC,
        colorbar=true,
        colorbar_title="TAC",
        xlabel="ISI",
        ylabel="Emission",
        title="2D Projection of Pareto Frontier (Color by TAC)",
        legend=false,
        markersize=5
    )
    display(p2d_all)
    
    # 6. Save results
    CSV.write("pareto_3d_results_new.csv", df_pareto)
    println("Saved $(nrow(df_pareto)) Pareto points to 'pareto_3d_results_new.csv'")

    return df_pareto,df_all
end


# Example of how to run the new function:
# This will perform a small run to test.
# For a full run, you can increase n_isi and n_inner_points.
println("\nStarting a test run of the 3D Pareto calculation...")
df_final_pareto, df_final_all = calculate_and_plot_3d_frontier(
    n_isi=20, 
    n_inner_points=20, 
    solver=:BARON
);

println("\nScript finished. Final Pareto points:")
println(df_final_pareto)
