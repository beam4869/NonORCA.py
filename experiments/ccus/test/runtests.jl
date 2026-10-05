using Test
using ForwardDiff
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "src", "CCUSModel.jl"))
using .CCUSModel

@testset "CCUS model structure" begin
    p = baseline_parameters()
    p_unforced = with_parameters(p; min_sink1_flow=0.0)
    z = initial_feasible_guess(p)
    @test length(z) == 82
    @test length(variable_names()) == 82
    @test length(equality_names()) == 34
    @test length(inequality_names()) == 172
    @test maximum(abs, equality_values(z, p)) <= 1.0e-12
    @test maximum(abs, source_carbon_residuals(z, p)) <= 1.0e-12
    @test maximum(inequality_values(z, p)) <= 1.0e-12
    @test objective_values(z, p).NetCapture >= p.capture_fraction * co2_generation(p)
    @test p_unforced.min_sink1_flow == 0.0
    @test p_unforced.Gmax_k == p.Gmax_k
    direct_use = direct_use_expansion_parameters()
    condition1 = supply_chain_condition1_parameters()
    @test direct_use.min_sink1_flow == 0.0
    @test direct_use.Zmin_k == Float64[0.06, 0.20, 0.95, 0.999, 0.999, 0.999]
    @test direct_use.Gmax_k == Float64[300, 1400, 1400, 756, 900, 913]
    @test direct_use.compressor_power[:, 5] == 0.18 .* p.compressor_power[:, 5]
    @test all(
        getfield(condition1, name) == getfield(direct_use, name)
        for name in fieldnames(CCUSParameters)
    )
    condition2 = supply_chain_condition2_parameters()
    @test condition2.M_s == Float64[357, 1050, 350, 2800]
    @test condition2.Gmax_k == Float64[250, 1000, 650, 500, 700, 500]
    @test condition2.Mu_k == Float64[0.30, 0.45, 0.02, 0.30, 0.00, 0.40]
    @test condition2.power_price == 0.08
    @test condition2.power_carbon_intensity == 0.732
    @test sink_names() == ["Algae", "Greenhouse", "Saline Storage", "Methanol", "Urea", "Acetic Acid"]
    structure = system_structure(z, p)
    @test isapprox(structure.pretreated_fraction + structure.direct_fraction, 1.0; atol=1.0e-12)
    @test structure.main_sink in 1:6
end

@testset "ForwardDiff agrees with central differences" begin
    p = baseline_parameters()
    z = fill(0.2, 82)
    columns = [1, 5, 29, 53, 77, 81, 82]
    h = 1.0e-6

    Jobj = ForwardDiff.jacobian(x -> objective_vector(x, p), z)
    Jeq = ForwardDiff.jacobian(x -> equality_values(x, p), z)
    Jineq = ForwardDiff.jacobian(x -> inequality_values(x, p), z)
    for j in columns
        zp = copy(z); zp[j] += h
        zm = copy(z); zm[j] -= h
        fd_obj = (objective_vector(zp, p) - objective_vector(zm, p)) / (2h)
        fd_eq = (equality_values(zp, p) - equality_values(zm, p)) / (2h)
        fd_ineq = (inequality_values(zp, p) - inequality_values(zm, p)) / (2h)
        @test isapprox(Jobj[:, j], fd_obj; rtol=2.0e-5, atol=2.0e-5)
        @test isapprox(Jeq[:, j], fd_eq; rtol=2.0e-5, atol=2.0e-7)
        @test isapprox(Jineq[:, j], fd_ineq; rtol=2.0e-5, atol=2.0e-7)
    end
end

@testset "Baseline solver smoke test" begin
    p = baseline_parameters()
    result = solve_ccus(p; objective=:feasibility, time_limit=60.0)
    @test result.termination in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
    @test result.primal_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
    @test result.residuals.max_abs_equality <= 1.0e-6
    @test result.residuals.max_abs_source_carbon_diagnostic <= 1.0e-6
    @test result.residuals.max_inequality_violation <= 1.0e-6
end
