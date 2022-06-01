using Distributed
addprocs(2)

@everywhere using Test
@everywhere using ExaComDecOPF
@everywhere using Ipopt

@everywhere ExaComDecOPF.load_case("../data/case9")
ExaComDecOPF.run_decomp(Ipopt.Optimizer; iterlim=100, rho_pq=400.0, rho_va=40000.0)
