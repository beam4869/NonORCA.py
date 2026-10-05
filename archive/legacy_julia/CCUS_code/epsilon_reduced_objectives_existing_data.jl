using JuMP, CPLEX, StatsBase, CSV, DataFrames, XLSX, LinearAlgebra, Ipopt, BARON, Plots,MathOptInterface
df1 = CSV.read("eps_Pareto.csv", DataFrame)
df2 = CSV.read("second_pareto.csv", DataFrame)
df_all = vcat(df1, df2)
p3d = scatter(
    df_all.Emiss, df_all.ISI, df_all.TAC,
    xlabel="Emission", ylabel="ISI", zlabel="TAC",
    title="3D Pareto Frontier",
    legend=false,
    markersize=4, markerstrokewidth=0
)
display(p3d)

p2d_all = scatter(
    df_all.ISI, df_all.Emiss,
    marker_z=df_all.TAC,
    colorbar=true,
    colorbar_title="TAC",
    xlabel="ISI",
    ylabel="Emission",
    # title="2D Projection of Pareto Frontier (Color by TAC)",
    legend=false,
    markersize=5
)
display(p2d_all)
p2d_Pareto = scatter(
    df_all.ISI*6000000+5*df_all.Emiss, df_all.TAC,
    xlabel="ISIx6000000+Emissionx5", xguidefontsize=12,
    ylabel="TAC", yguidefontsize=12,
    title="Objective Reduction Pareto Frontier",
    legend=false,
    markersize=6
)
display(p2d_Pareto)