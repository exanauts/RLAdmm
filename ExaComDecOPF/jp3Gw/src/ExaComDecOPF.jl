module ExaComDecOPF

using Distributed
using DelimitedFiles
using JuMP
using Printf
using FileIO
using Statistics
using LinearAlgebra

include("opfdata.jl")
#include("acopf_model.jl")
#include("acopf_admm_polar.jl")
include("acopf_admm_rect.jl")

data = nothing
ybus = nothing
case = nothing
brmod = nothing
ll_idx = nothing

function run_enitre(case)
    data = opf_loaddata(case)
    ybus = Ybus(computeAdmitances(data.lines, data.buses, data.baseMVA)...)
    m = acopf_model(data, ybus)
    setsolver(m, IpoptSolver())
    solve(m)
end

function run_decomp(optimizer_constructor; iterlim=100, rho_pq=400.0, rho_va=40000.0, eps=1e-4, Kf=100, eta=0.99,
                    savesol=false, loadsol=false, update_rho=true, use_auglag=false, use_whole=false, linelimit=true)
    return admm_rect(optimizer_constructor; casename=basename(case), iterlim=iterlim, rho_pq=rho_pq, rho_va=rho_va, eps=eps, Kf=Kf, eta=eta,
                     savesol=savesol, loadsol=loadsol, update_rho=update_rho, use_auglag=use_auglag, use_whole=use_whole, linelimit=linelimit)
end

function restart_decomp(param, info, tau; iterlim=100, log=true, final_message=true, update_rho=false, increase_tau=false)
    return admm_rect_restart(param, info, tau; iterlim=iterlim, log=log, final_message=final_message, update_rho=update_rho, increase_tau=increase_tau)
end

function load_case(case_; perturb=false)
    global case = case_
    global data = opf_loaddata(case)
    global ybus = Ybus(computeAdmitances(data.lines, data.buses, data.baseMVA)...)
    if perturb
        perturb_load(0.1)
    end
end


function perturb_load(magnitude)
    num_buses = size(data.buses,1)
    for i in 1:num_buses
        data.buses[i].Pd = data.buses[i].Pd*(1+magnitude-2*magnitude*rand())
        data.buses[i].Qd = data.buses[i].Qd*(1+magnitude-2*magnitude*rand())
    end
end



end