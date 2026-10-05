############# PACKAGES #############
using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX, LinearAlgebra, Ipopt, BARON, Plots

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
        # Create the model first
        model = Model(BARON.Optimizer)
        # Then set attributes
        set_optimizer_attribute(model, "maxtime", 60)
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
function solve_once(objective::Symbol; eps_ISI::Union{Nothing,Float64}=nothing,
                                     eps_Emiss::Union{Nothing,Float64}=nothing,
                                     eps_Cost::Union{Nothing,Float64}=nothing,
                                     solver=:BARON)
    M = build_model(; solver)
    m, TAC, TotEmiss, InherSafeIndic = M.m, M.TAC, M.TotEmiss, M.InherSafeIndic

    # ε-constraints
    if eps_ISI !== nothing
        @NLconstraint(m, InherSafeIndic <= eps_ISI)
    end
    if eps_Emiss !== nothing
        @constraint(m, TotEmiss <= eps_Emiss)
    end
    if eps_Cost !== nothing
        @NLconstraint(m, TAC <= eps_Cost)
    end

    # Objective choice
    if objective == :emiss
        @NLobjective(m, Min, TotEmiss)
    elseif objective == :isi
        @NLobjective(m, Min, InherSafeIndic)
    elseif objective == :cost
        @NLobjective(m, Min, TAC)
    else
        error("Unknown objective $(objective)")
    end

    optimize!(m)

    st = termination_status(m)
    
    # Only try to read values if a solution was found
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
        return (status = st,
                TAC    = value(TAC),
                Emiss  = value(TotEmiss),
                ISI    = value(InherSafeIndic))
    else
        # Return NaN for values if no solution was found
        return (status = st,
                TAC    = NaN,
                Emiss  = NaN,
                ISI    = NaN)
    end
end

############# FRONTIER DRIVER + PLOT #############
"""
    epsilon_frontier(; Neps=50, solver=:BARON, do_dual_sweep=true, cost_cap=nothing)

Computes an ε-constraint frontier. Primary sweep: Min Emissions s.t. ISI ≤ ε.
If `do_dual_sweep=true`, also Min ISI s.t. Emissions ≤ ε and plot both.
If `cost_cap` is provided, adds TAC ≤ cost_cap to each run.

Returns a NamedTuple with arrays (x, y, c) and displays the plot.
"""
function epsilon_frontier(; Neps::Int=50, solver=:BARON, do_dual_sweep::Bool=true,
                           cost_cap::Union{Nothing,Float64}=nothing)

    # Anchors
    best_emiss = solve_once(:emiss; solver)     # min Emissions (GWP)
    best_isi   = solve_once(:isi;   solver)     # min ISI

    # ε range on ISI for the primary sweep
    eps_lo = best_isi.ISI
    eps_hi = best_emiss.ISI
    eps_grid = collect(range(eps_lo, eps_hi; length=Neps))

    # Primary sweep: Min Emissions s.t. ISI ≤ ε
    F1 = NamedTuple[]
    for eps in eps_grid
        push!(F1, solve_once(:emiss; eps_ISI=eps, eps_Cost=cost_cap, solver))
    end

    # Optional second sweep: Min ISI s.t. Emissions ≤ ε
    F2 = NamedTuple[]
    if do_dual_sweep
        eps2_lo = best_emiss.Emiss
        eps2_hi = best_isi.Emiss
        eps2_grid = collect(range(eps2_lo, eps2_hi; length=Neps))
        for eps in eps2_grid
            push!(F2, solve_once(:isi; eps_Emiss=eps, eps_Cost=cost_cap, solver))
        end
    end

    # Collect for plotting
    x = [p.Emiss for p in F1]
    y = [p.ISI   for p in F1]
    c = [p.TAC   for p in F1]
    pal = cgrad(:grays, rev=true)

    plt = scatter(
    x, y;
    marker_z = c,         # <-- color-mapped values
    color    = pal,       # <-- gradient/palette, not a single color
    colorbar = true,
    colorbar_title = "Cost",
    xlabel = "Global Warming Potential (Emissions)",
    ylabel = "Worker Injuries (ISI)",
    markersize = 6,
    legend = false,
    title = "ε-Constraint Pareto Frontier (Emissions vs ISI, colored by TAC)",
    )


    if do_dual_sweep && !isempty(F2)
        x2 = [p.Emiss for p in F2]
        y2 = [p.ISI   for p in F2]
        c2 = [p.TAC   for p in F2]
        scatter!(x2, y2; marker_z=c2, color=pal, markersize=6)
    end

    display(plt)
    return (x=x, y=y, c=c, anchors=(best_emiss=best_emiss, best_isi=best_isi))
end

############# RUN #############
# Call this to compute & plot:
# result = epsilon_frontier(; Neps=60, solver=:BARON, do_dual_sweep=true)
# The returned `result` has arrays (x, y, c) and anchor-point info.
######################## 3-OBJ PARETO (project & highlight) ########################
# Samples the 3D space from three ε-constraint perspectives, Pareto-filters in 3D,
# then plots (Emissions, ISI) with Cost mapped to gray and the 3D Pareto set in red.

# 3D nondominated filter (minimize all)
function _nondominated_3d(points::Vector{<:NamedTuple})
    keep = trues(length(points))
    @inbounds for i in eachindex(points)
        keep[i] || continue
        pi = points[i]
        # Filter out points with NaN values
        if isnan(pi.TAC) || isnan(pi.Emiss) || isnan(pi.ISI)
            keep[i] = false
            continue
        end
        for j in eachindex(points)
            (i == j || !keep[j]) && continue
            pj = points[j]
            if isnan(pj.TAC) || isnan(pj.Emiss) || isnan(pj.ISI)
                continue
            end
            if (pj.TAC   <= pi.TAC   &&
                pj.Emiss <= pi.Emiss &&
                pj.ISI   <= pi.ISI   &&
               (pj.TAC   <  pi.TAC   ||
                pj.Emiss <  pi.Emiss ||
                pj.ISI   <  pi.ISI))
                keep[i] = false
                break
            end
        end
    end
    return points[keep]
end

# Anchors to define ε ranges
function _anchors(; solver=:BARON)
    return (cost  = solve_once(:cost;  solver),
            emiss = solve_once(:emiss; solver),
            isi   = solve_once(:isi;   solver))
end

"""
    pareto3_project_and_highlight(; N1=18, N2=18, solver=:BARON, alpha_bg=0.7)

Compute a dense 3-objective Pareto set via ε-constraint sampling from three perspectives:
  (i)  min Cost    s.t. Emiss ≤ ε₁, ISI ≤ ε₂
  (ii) min Emiss   s.t. Cost  ≤ ε₁, ISI ≤ ε₂
  (iii)min ISI     s.t. Cost  ≤ ε₁, Emiss ≤ ε₂

Then:
  • plots all sampled feasible points: (Emissions, ISI), color = Cost (gray, high→dark)
  • overlays 3D nondominated points in red (the true 3-obj Pareto set)

Returns: (all_pts, pareto3d_pts, fig)
"""
function pareto3_project_and_highlight(; N1::Int=18, N2::Int=18, solver=:BARON, alpha_bg=0.7)

    A = _anchors(; solver)
    # Build loose ε-ranges with tiny buffers
    buf = -1e-1
    e_lo, e_hi = (min(A.emiss.Emiss, A.isi.Emiss)*(1-buf),
                  max(A.emiss.Emiss, A.isi.Emiss)*(1+buf))
    i_lo, i_hi = (min(A.isi.ISI, A.emiss.ISI)*(1-buf),
                  max(A.isi.ISI, A.emiss.ISI)*(1+buf))
    c_lo, c_hi = (A.cost.TAC*(1-buf),
                  max(A.emiss.TAC, A.isi.TAC, A.cost.TAC)*(1+buf))

    epsE = collect(range(e_lo, e_hi; length=N1))
    epsI = collect(range(i_lo, i_hi; length=N2))
    epsC = collect(range(c_lo, c_hi; length=N1))

    # Sample from three perspectives
    pts = NamedTuple[]

    # (i) min Cost  | Emiss ≤ ε, ISI ≤ ε
    for e in epsE, i in epsI
        r = solve_once(:cost;  eps_Emiss=e, eps_ISI=i, solver)
        push!(pts, (TAC=r.TAC, Emiss=r.Emiss, ISI=r.ISI))
    end
    # (ii) min Emiss | Cost ≤ ε, ISI ≤ ε
    for c in epsC, i in epsI
        r = solve_once(:emiss; eps_Cost=c, eps_ISI=i, solver)
        push!(pts, (TAC=r.TAC, Emiss=r.Emiss, ISI=r.ISI))
    end
    # (iii) min ISI  | Cost ≤ ε, Emiss ≤ ε
    for c in epsC, e in epsE
        r = solve_once(:isi;   eps_Cost=c, eps_Emiss=e, solver)
        push!(pts, (TAC=r.TAC, Emiss=r.Emiss, ISI=r.ISI))
    end

    # Light de-dup to compress near-identical solutions
    rd(x) = round(x; digits=6)
    pts = unique([(TAC=rd(p.TAC), Emiss=rd(p.Emiss), ISI=rd(p.ISI)) for p in pts])

    # True 3D Pareto set
    pts_nd = _nondominated_3d(pts)

    # Plot: background cloud (gray, colored by Cost), then red Pareto overlay
    x_all = [p.Emiss for p in pts]
    y_all = [p.ISI   for p in pts]
    z_all = [p.TAC   for p in pts]

    pal = cgrad(:grays, rev=true)  # dark = high cost (matches paper’s style)

    fig = scatter(
        x_all, y_all;
        marker_z = z_all, color = pal,
        colorbar_title = "Cost",
        xlabel = "Global Warming Potential (Emissions)",
        ylabel = "Worker Injuries (ISI)",
        title  = "3-Objective Pareto (projected): Cost on (Emissions, ISI)",
        markersize = 5, legend = false, alpha = alpha_bg,
    )

    # Red overlay = 3D nondominated projection
    x_nd = [p.Emiss for p in pts_nd]
    y_nd = [p.ISI   for p in pts_nd]
    scatter!(x_nd, y_nd; color=:red, markerstrokecolor=:red, markersize=4)

    display(fig)
    return (all_pts = pts, pareto3d_pts = pts_nd, fig = fig)
end

# Example:
res = pareto3_project_and_highlight(; N1=2, N2=2, solver=:BARON, alpha_bg=0.6)
