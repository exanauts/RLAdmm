using Distributed
addprocs(1)
nprocs()

@everywhere using ExaComDecOPF
@everywhere using Ipopt
# using Flux
using JuMP
# using Distributions
# using ProgressMeter
# using DataFrames
# using CSV
# using Plots
# using Statistics
using Random
using Printf
using LaTeXStrings
using BSON: @save

# Identify solver for ADMM
solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0, "sb"=>"yes")

@everywhere ExaComDecOPF.load_case("/users/Sihan/Documents/Argonne/data/case9")


function environment(t, action_pq, action_va, param, info, tau; update_rho=false, perturb=false, reload=false, param_reload=nothing)

    # Scale the actions
    # Take absolute value so that resulting rho values are always positive
    # action_pq_scaled = abs.(Float64(action_pq).*1.0)
    # action_va_scaled = abs.(Float64(action_va).*1.0)
    action_pq_scaled = abs.(action_pq.*1.0)
    action_va_scaled = abs.(action_va.*1.0)
    
    #println("action_pq_scaled: $action_pq_scaled")
    #println("action_va_scaled: $action_va_scaled")
    #for the first time step
    if t==0
        # Take the first step in ADMM
        if perturb
            @everywhere ExaComDecOPF.load_case("/users/Sihan/Documents/Argonne/data/case9"; perturb=true)
        end
        if reload
#             @everywhere ExaComDecOPF.load_case("/users/Sihan/Documents/Argonne/data/case9_cutline")
            @everywhere ExaComDecOPF.load_case("/users/Sihan/Documents/Argonne/data/case9_removegen")
        end
#         param, info, tau = ExaComDecOPF.run_decomp(solver; iterlim=1, rho_pq=action_pq_scaled, rho_va=action_va_scaled)
        param, info, tau = ExaComDecOPF.run_decomp(solver; iterlim=1, rho_pq=action_pq_scaled[1], rho_va=action_va_scaled[1])
#         if reload
#             param.u_curr = param_reload.u_curr
#             param.v_curr = param_reload.v_curr
#         end
    else
        
        # Assign the new rho values
        #param.rho_pq = action_pq_scaled
        #param.rho_va = action_va_scaled
        
        nvars = 2*param.ngen + 6*param.nline
#         println(param.ngen)
#         println(param.nline)
        param.rho = zeros(nvars)
        param.rho[1:2*param.ngen+4*param.nline] .= action_pq_scaled
        param.rho[2*param.ngen+4*param.nline+1:end] .= action_va_scaled

        
#         println("action_pq_scaled: $action_pq_scaled")
#         println("action_va_scaled: $action_va_scaled")
        
        param_copy = deepcopy(param)
        info_copy = deepcopy(info)
        tau_copy = deepcopy(tau)

        # Take another step in ADMM using rho values
        param_copy, info_copy, tau_copy= ExaComDecOPF.restart_decomp(param_copy, info_copy, tau_copy; iterlim=1, log=false, final_message=false, update_rho=update_rho, increase_tau=update_rho)
#         param_copy, info_copy, tau_copy= ExaComDecOPF.restart_decomp(param_copy, info_copy, tau_copy; iterlim=1, log=false, final_message=false)

# #         ExaComDecOPF.restart_decomp(param, info, tau; iterlim=1, log=false, final_message=false)
#         param, info, tau = ExaComDecOPF.restart_decomp(param, info, tau; iterlim=1, log=false, final_message=false)
# #         param, info, tau = ExaComDecOPF.restart_decomp(param, info, tau; iterlim=1, log=true, final_message=false)
# #         println(info.primres, info_copy.primres)
#         return param, info, tau
    end
    
    # Identify the new state
    if t==0
        info_copy = info
    end
    new_state = [info_copy.primres, info_copy.dualres]
#     if t==0
#         new_state = [info.primres, info.dualres]
#     else
#         new_state = [info_copy.primres, info_copy.dualres]
#     end
#     # Determine the reward associated with the new state
#     iter_reward1, iter_reward2, converge, prim_thres, dual_thres = reward(new_state, info)
    
#     return new_state, iter_reward1, iter_reward2, converge, prim_thres, dual_thres, param, info, tau, action_pq_scaled, action_va_scaled

    
    if new_state[1] < info_copy.eps_pri && new_state[2] < info_copy.eps_dual
        converge = true
        prim_thres = true
        dual_thres = true
    else
        converge = false
        if new_state[1] < info_copy.eps_pri
            prim_thres = true
            dual_thres = false
        elseif new_state[2] < info_copy.eps_dual
            prim_thres = false
            dual_thres = true
        else
            prim_thres = false
            dual_thres = false
        end
    end
    
    if t==0
        return new_state, converge, prim_thres, dual_thres, param, info, tau
    else
        return new_state, converge, prim_thres, dual_thres, param_copy, info_copy, tau_copy
    end
end