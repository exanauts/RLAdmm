mutable struct AdmmParamRect
    rho_pq::Float64
    rho_va::Float64
    rho::Vector{Float64}
    rho_max::Float64
    rho_min_pq::Float64
    rho_min_w::Float64
    eps_rp::Float64
    eps_rp_min::Float64
    eta::Float64
    rt_inc::Float64
    rt_dec::Float64
    rt_inc2::Float64
    rt_dec2::Float64
    Kf::Int
    Kf_mean::Int

    use_auglag::Bool
    use_whole::Bool
    linelimit::Bool
    ngen::Int
    nline::Int
    nbus::Int
    pg_start::Int
    qg_start::Int
    pij_start::Int
    qij_start::Int
    pji_start::Int
    qji_start::Int
    wi_i_ij_start::Int
    wi_j_ji_start::Int
    u_curr::Vector{Float64}
    u_prev::Vector{Float64}
    v_curr::Vector{Float64}
    v_prev::Vector{Float64}
    l_curr::Vector{Float64}
    l_prev::Vector{Float64}
    delta_u::Vector{Float64}
    delta_v::Vector{Float64}
    delta_l::Vector{Float64}
    alpha::Vector{Float64}
    beta::Vector{Float64}
    rd::Vector{Float64}
    rp::Vector{Float64}
    rp_old::Vector{Float64}
    rp_k0::Vector{Float64}

    casename::String
    optimizer_constructor

    function AdmmParamRect(ngen, nline, nbus, rho_pq, rho_va;
                           use_auglag=false, use_whole=false, linelimit=true)
        param = new()
        nvars = 2*ngen + 6*nline
        param.use_auglag = use_auglag
        param.use_whole = use_whole
        param.linelimit = linelimit
        param.ngen = ngen
        param.nline = nline
        param.nbus = nbus
        param.pg_start = 0
        param.qg_start = ngen
        param.pij_start = 2*ngen
        param.qij_start = 2*ngen + nline
        param.pji_start = 2*ngen + 2*nline
        param.qji_start = 2*ngen + 3*nline
        param.wi_i_ij_start = 2*ngen + 4*nline
        param.wi_j_ji_start = 2*ngen + 5*nline
        param.rho_pq = rho_pq
        param.rho_va = rho_va
        param.rho = zeros(nvars)
        param.rho[1:2*ngen+4*nline] .= rho_pq
        param.rho[2*ngen+4*nline+1:end] .= rho_va
        param.rho_max = 1e6
        param.rho_min_pq = 5.0
        param.rho_min_w = 5.0
        param.eps_rp = 1e-4
        param.eps_rp_min = 1e-5
        param.eta = 0.99
        param.rt_inc = 2
        param.rt_dec = 2
        param.rt_inc2 = 1.5
        param.rt_dec2 = 1.5
        param.Kf = 100
        param.Kf_mean = 10
        param.u_curr = zeros(nvars)
        param.u_prev = zeros(nvars)
        param.v_curr = zeros(nvars)
        param.v_prev = zeros(nvars)
        param.l_curr = zeros(nvars)
        param.l_prev = zeros(nvars)
        param.delta_u = zeros(nvars)
        param.delta_v = zeros(nvars)
        param.delta_l = zeros(nvars)
        param.alpha = zeros(nvars)
        param.beta = zeros(nvars)
        param.rd = zeros(nvars)
        param.rp = zeros(nvars)
        param.rp_old = zeros(nvars)
        param.rp_k0 = zeros(nvars)
        param.optimizer_constructor = nothing
        return param
    end
end

mutable struct AdmmInfo
    it::Int
    br_infeas::Int
    br_refail::Int
    br_other::Int
    term::Symbol
    obj::Float64
    gmax::Float64
    primres::Float64
    dualres::Float64
    rho_pq_max::Float64
    rho_pq_min::Float64
    rho_va_max::Float64
    rho_va_min::Float64
    eps_pri::Float64
    eps_dual::Float64
    avg_gen::Float64
    avg_br::Float64
    avg_bus::Float64
    avg_auglag::Float64
    cum_avg_gen::Float64
    cum_avg_br::Float64
    cum_avg_bus::Float64
    elapsed::Float64

    function AdmmInfo()
        info = new()
        info.it = 0
        info.br_infeas = 0
        info.br_refail = 0
        info.br_other = 0
        info.term = :IterLimit
        info.obj = 0
        info.gmax = Inf
        info.primres = Inf
        info.dualres = Inf
        info.rho_pq_max = -Inf
        info.rho_pq_min = Inf
        info.rho_va_max = -Inf
        info.rho_va_min = Inf
        info.eps_pri = Inf
        info.eps_dual = Inf
        info.avg_gen = 0
        info.avg_br = 0
        info.avg_bus = 0
        info.avg_auglag = 0
        info.cum_avg_gen = 0
        info.cum_avg_br = 0
        info.cum_avg_bus = 0
        info.elapsed = 0
        return info
    end
end

function get_start(p, nele, nrem)
    return (p-2)*nele + 1 + (((p-1) <= nrem) ? (p-2) : nrem)
end

function get_inc(p, nele, nrem)
    return nele - 1 + (((p-1) <= nrem) ? 1 : 0)
end

function solve_generators(param::AdmmParamRect, info::AdmmInfo)
    baseMVA = data.baseMVA
    gens = data.generators
    ngen = length(data.generators)

    p = myid()
    nele = floor(Int, ngen / nworkers())
    nrem = mod(ngen, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    u_curr = param.u_curr
    v_curr = param.v_curr
    l_curr = param.l_curr

    t = 0.0
    for g=start:last
        # Compute a closed-form solution.
        t += @elapsed begin
            u_curr[param.pg_start+g] = max(gens[g].Pmin,
                              min(gens[g].Pmax,
                                  (-(gens[g].coeff[gens[g].n-1]*baseMVA + l_curr[param.pg_start+g] -
                                     param.rho[param.pg_start+g]*v_curr[param.pg_start+g])) / (2*gens[g].coeff[gens[g].n-2]*(baseMVA^2) + param.rho[param.pg_start+g])))
            u_curr[param.qg_start+g] = max(gens[g].Qmin, min(gens[g].Qmax, (-(l_curr[param.qg_start+g] - param.rho[param.qg_start+g]*v_curr[param.qg_start+g])) / param.rho[param.qg_start+g]))
        end
    end

    if start <= last
        info.avg_gen = t / (last-start+1)
        info.cum_avg_gen += info.avg_gen
    end

    return param, info
end

function build_whole_branch_models(param::AdmmParamRect)
    baseMVA = data.baseMVA
    BusIdx = data.BusIdx
    lines = data.lines
    buses = data.buses

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    nline = length(data.lines)
    p = myid()
    nele = floor(Int, nline / nworkers())
    nrem = mod(nline, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    global brmod = Vector{JuMP.Model}(undef, 1)
    global ll_idx = findall(l -> l.rateA > 0 && l.rateA < 1e10, lines[start:last])

    m = Model(param.optimizer_constructor)

    n = last - start + 1
    @variable(m, pij[l=1:n])
    @variable(m, qij[l=1:n])
    @variable(m, buses[BusIdx[lines[start+l-1].from]].Vmin^2 <= wi_i_ij[l=1:n] <= buses[BusIdx[lines[start+l-1].from]].Vmax^2)
    @variable(m, pji[l=1:n])
    @variable(m, qji[l=1:n])
    @variable(m, buses[BusIdx[lines[start+l-1].to]].Vmin^2 <= wi_j_ji[l=1:n] <= buses[BusIdx[lines[start+l-1].to]].Vmax^2)
    @variable(m, wRij[l=1:n])
    @variable(m, wIij[l=1:n])

    if param.use_auglag
        @NLexpression(m, flow_pij[l=1:n], pij[l] - (YffR[start+l-1]*wi_i_ij[l] + YftR[start+l-1]*wRij[l] + YftI[start+l-1]*wIij[l]))
        @NLexpression(m, flow_qij[l=1:n], qij[l] - (-YffI[start+l-1]*wi_i_ij[l] - YftI[start+l-1]*wRij[l] + YftR[start+l-1]*wIij[l]))
        @NLexpression(m, flow_pji[l=1:n], pji[l] - (YttR[start+l-1]*wi_j_ji[l] + YtfR[start+l-1]*wRij[l] - YtfI[start+l-1]*wIij[l]))
        @NLexpression(m, flow_qji[l=1:n], qji[l] - (-YttI[start+l-1]*wi_j_ji[l] - YtfI[start+l-1]*wRij[l] - YtfR[start+l-1]*wIij[l]))
        @NLexpression(m, rect[l=1:n], wRij[l]^2 + wIij[l]^2 - wi_i_ij[l]*wi_j_ji[l])
        m[:l_flow_pij] = zeros(n)
        m[:l_flow_qij] = zeros(n)
        m[:l_flow_pji] = zeros(n)
        m[:l_flow_qji] = zeros(n)
        m[:l_rect] = zeros(n)
        m[:l_line_sij] = zeros(n)
        m[:l_line_sji] = zeros(n)
        m[:mu] = 10.0

        @NLexpression(m, line_sij[l=1:n], 0.0)
        @NLexpression(m, line_sji[l=1:n], 0.0)

        if param.linelimit && !isempty(ll_idx)
            @variable(m, sij[l=1:length(ll_idx)] >= 0)
            @variable(m, sji[l=1:length(ll_idx)] >= 0)
            # Manually reassign the NLexpresion to avoid JuMP's dictionary check.
            for l=1:length(ll_idx)
                m[:line_sij][l] = @NLexpression(m, pij[ll_idx[l]]^2 + qij[ll_idx[l]]^2 - (lines[start+ll_idx[l]-1].rateA/baseMVA)^2 + sij[l])
                m[:line_sji][l] = @NLexpression(m, pji[ll_idx[l]]^2 + qji[ll_idx[l]]^2 - (lines[start+ll_idx[l]-1].rateA/baseMVA)^2 + sji[l])
            end
        end
    else
        @constraint(m, flow_pij[l=1:n], pij[l] == YffR[start+l-1]*wi_i_ij[l] + YftR[start+l-1]*wRij[l] + YftI[start+l-1]*wIij[l])
        @constraint(m, flow_qij[l=1:n], qij[l] == -YffI[start+l-1]*wi_i_ij[l] - YftI[start+l-1]*wRij[l] + YftR[start+l-1]*wIij[l])
        @constraint(m, flow_pji[l=1:n], pji[l] == YttR[start+l-1]*wi_j_ji[l] + YtfR[start+l-1]*wRij[l] - YtfI[start+l-1]*wIij[l])
        @constraint(m, flow_qji[l=1:n], qji[l] == -YttI[start+l-1]*wi_j_ji[l] - YtfI[start+l-1]*wRij[l] - YtfR[start+l-1]*wIij[l])
        @NLconstraint(m, rect[l=1:n], wRij[l]^2 + wIij[l]^2 == wi_i_ij[l]*wi_j_ji[l])

        if param.linelimit && !isempty(ll_idx)
            @NLconstraint(m, line_sij[l=1:length(ll_idx)], pij[ll_idx[l]]^2 + qij[ll_idx[l]]^2 <= (lines[start+ll_idx[l]-1].rateA/baseMVA)^2)
            @NLconstraint(m, line_sji[l=1:length(ll_idx)], pji[ll_idx[l]]^2 + qji[ll_idx[l]]^2 <= (lines[start+ll_idx[l]-1].rateA/baseMVA)^2)
        end
    end
    brmod[1] = m
end

function initialize_branch_models(param::AdmmParamRect)
    if param.use_whole
        return initialize_whole_branch_models(param)
    end
end

function initialize_whole_branch_models(param::AdmmParamRect)
    baseMVA = data.baseMVA
    BusIdx = data.BusIdx
    lines = data.lines
    buses = data.buses

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    u_curr = param.u_curr
    for l=1:length(data.lines)
        wij0 = (buses[BusIdx[lines[l].from]].Vmax^2 + buses[BusIdx[lines[l].from]].Vmin^2) / 2
        wji0 = (buses[BusIdx[lines[l].to]].Vmax^2 + buses[BusIdx[lines[l].to]].Vmin^2) / 2
        wR0 = sqrt(wij0 * wji0)
        u_curr[param.pij_start+l] = YffR[l] * wij0 + YftR[l] * wR0
        u_curr[param.qij_start+l] = -YffI[l] * wij0 - YftI[l] * wR0
        u_curr[param.wi_i_ij_start+l] = wij0
        u_curr[param.pji_start+l] = YttR[l] * wji0 + YtfR[l] * wR0
        u_curr[param.qji_start+l] = -YttI[l] * wji0 - YtfI[l] * wR0
        u_curr[param.wi_j_ji_start+l] = wji0
        # @show l, u_curr[param.pij_start+l], u_curr[param.qij_start+l], u_curr[param.wi_i_ij_start+l], u_curr[param.pji_start+l], u_curr[param.qji_start+l], u_curr[param.wi_j_ji_start+l]
    end
end

function build_branch_models(param::AdmmParamRect)
    if param.use_whole
        return build_whole_branch_models(param)
    end

    baseMVA = data.baseMVA
    BusIdx = data.BusIdx
    lines = data.lines
    buses = data.buses

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    nline = length(data.lines)
    p = myid()
    nele = floor(Int, nline / nworkers())
    nrem = mod(nline, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    global brmod = Vector{JuMP.Model}(undef, last - start + 1)
    for l=start:last
        fr = BusIdx[lines[l].from]
        to = BusIdx[lines[l].to]

        m = Model(param.optimizer_constructor)

        @variable(m, pij)
        @variable(m, qij)
        @variable(m, buses[fr].Vmin^2 <= wi_i_ij <= buses[fr].Vmax^2)
        @variable(m, pji)
        @variable(m, qji)
        @variable(m, buses[to].Vmin^2 <= wi_j_ji <= buses[to].Vmax^2)
        @variable(m, wRij)
        @variable(m, wIij)

        if param.use_auglag
            if JuMP.solver_name(m) == "Tron"
                set_optimizer_attribute(m, "tol", 1e-8)
            end

            @NLexpression(m, flow_pij, pij - (YffR[l]*wi_i_ij + YftR[l]*wRij + YftI[l]*wIij))
            @NLexpression(m, flow_qij, qij - (-YffI[l]*wi_i_ij - YftI[l]*wRij + YftR[l]*wIij))
            @NLexpression(m, flow_pji, pji - (YttR[l]*wi_j_ji + YtfR[l]*wRij - YtfI[l]*wIij))
            @NLexpression(m, flow_qji, qji - (-YttI[l]*wi_j_ji - YtfI[l]*wRij - YtfR[l]*wIij))
            @NLexpression(m, rect, wRij^2 + wIij^2 - wi_i_ij*wi_j_ji)

            m[:l_flow_pij] = 0
            m[:l_flow_qij] = 0
            m[:l_flow_pji] = 0
            m[:l_flow_qji] = 0
            m[:l_rect] = 0
            m[:l_line_sij] = 0
            m[:l_line_sji] = 0
            m[:mu] = 10

            @NLexpression(m, line_sij, 0)
            @NLexpression(m, line_sji, 0)

            if param.linelimit
                if lines[l].rateA > 0 && lines[l].rateA < 1e10
                    @variable(m, sij >= 0)
                    @variable(m, sji >= 0)
                    m[:line_sij] = @NLexpression(m, pij^2 + qij^2 - (lines[l].rateA/baseMVA)^2 + sij)
                    m[:line_sji] = @NLexpression(m, pji^2 + qji^2 - (lines[l].rateA/baseMVA)^2 + sji)
                end
            end
        else
            @constraint(m, pij == YffR[l]*wi_i_ij + YftR[l]*wRij + YftI[l]*wIij)
            @constraint(m, qij == -YffI[l]*wi_i_ij - YftI[l]*wRij + YftR[l]*wIij)
            @constraint(m, pji == YttR[l]*wi_j_ji + YtfR[l]*wRij - YtfI[l]*wIij)
            @constraint(m, qji == -YttI[l]*wi_j_ji - YtfI[l]*wRij - YtfR[l]*wIij)
            @NLconstraint(m, wRij^2 + wIij^2 == wi_i_ij*wi_j_ji)

            if param.linelimit
                if lines[l].rateA > 0 && lines[l].rateA < 1e10
                    @NLconstraint(m, pij^2 + qij^2 <= (lines[l].rateA / baseMVA)^2)
                    @NLconstraint(m, pji^2 + qji^2 <= (lines[l].rateA / baseMVA)^2)
                end
            end
        end

        brmod[l - start + 1] = m
    end
end

function solve_whole_branches(param::AdmmParamRect, info::AdmmInfo)
    baseMVA = data.baseMVA
    buses = data.buses
    lines = data.lines
    busref = data.bus_ref
    nline = length(data.lines)
    BusIdx = data.BusIdx

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    p = myid()
    nele = floor(Int, nline / nworkers())
    nrem = mod(nline, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    u_curr = param.u_curr
    v_curr = param.v_curr
    l_curr = param.l_curr
    rho = param.rho

    t = 0.0
    auglag_it = 0
    info.br_infeas = info.br_refail = info.br_other = 0

    m = brmod[1]
    n = last - start + 1
    for l=1:n
        set_start_value(m[:pij][l], u_curr[param.pij_start+start+l-1])
        set_start_value(m[:qij][l], u_curr[param.qij_start+start+l-1])
        set_start_value(m[:wi_i_ij][l], u_curr[param.wi_i_ij_start+start+l-1])
        set_start_value(m[:pji][l], u_curr[param.pji_start+start+l-1])
        set_start_value(m[:qji][l], u_curr[param.qji_start+start+l-1])
        set_start_value(m[:wi_j_ji][l], u_curr[param.wi_j_ji_start+start+l-1])
        set_start_value(m[:wRij][l], sqrt(u_curr[param.wi_i_ij_start+start+l-1] * u_curr[param.wi_j_ji_start+start+l-1]))
        set_start_value(m[:wIij][l], 0.0)
        # @show l, u_curr[param.pij_start+start+l-1], u_curr[param.qij_start+start+l-1], u_curr[param.wi_i_ij_start+start+l-1], u_curr[param.pji_start+start+l-1], u_curr[param.qji_start+start+l-1], u_curr[param.wi_j_ji_start+start+l-1]
    end

    if param.use_auglag
        if param.linelimit && !isempty(ll_idx)
            for l=1:length(ll_idx)
                set_start_value(m[:sij][l], (lines[start+ll_idx[l]-1].rateA/baseMVA)^2 - start_value(m[:pij][ll_idx[l]])^2 - start_value(m[:qij][ll_idx[l]])^2)
                set_start_value(m[:sji][l], (lines[start+ll_idx[l]-1].rateA/baseMVA)^2 - start_value(m[:pji][ll_idx[l]])^2 - start_value(m[:qji][ll_idx[l]])^2)
            end
        end

        it = 0
        terminate = false
        omega = 1 / m[:mu]
        eta = 1 / (m[:mu])^0.1
        cviol = zeros(7*n)

        while !terminate
            it += 1
            @NLobjective(m, Min,
                sum(l_curr[param.pij_start+start+l-1]*m[:pij][l] +
                    l_curr[param.qij_start+start+l-1]*m[:qij][l] +
                    l_curr[param.wi_i_ij_start+start+l-1]*m[:wi_i_ij][l] +
                    l_curr[param.pji_start+start+l-1]*m[:pji][l] +
                    l_curr[param.qji_start+start+l-1]*m[:qji][l] +
                    l_curr[param.wi_j_ji_start+start+l-1]*m[:wi_j_ji][l] +
                    0.5*rho[param.wi_i_ij_start+start+l-1]*(m[:wi_i_ij][l] - v_curr[param.wi_i_ij_start+start+l-1])^2 +
                    0.5*rho[param.wi_j_ji_start+start+l-1]*(m[:wi_j_ji][l] - v_curr[param.wi_j_ji_start+start+l-1])^2 +
                    0.5*rho[param.pij_start+start+l-1]*(m[:pij][l] - v_curr[param.pij_start+start+l-1])^2 +
                    0.5*rho[param.qij_start+start+l-1]*(m[:qij][l] - v_curr[param.qij_start+start+l-1])^2 +
                    0.5*rho[param.pji_start+start+l-1]*(m[:pji][l] - v_curr[param.pji_start+start+l-1])^2 +
                    0.5*rho[param.qji_start+start+l-1]*(m[:qji][l] - v_curr[param.qji_start+start+l-1])^2 +
                    m[:l_flow_pij][l]*m[:flow_pij][l] +
                    m[:l_flow_qij][l]*m[:flow_qij][l] +
                    m[:l_flow_pji][l]*m[:flow_pji][l] +
                    m[:l_flow_qji][l]*m[:flow_qji][l] +
                    m[:l_rect][l]*m[:rect][l] +
                    m[:l_line_sij][l]*m[:line_sij][l] +
                    m[:l_line_sji][l]*m[:line_sji][l] +
                    0.5*m[:mu]*(m[:flow_pij][l])^2 +
                    0.5*m[:mu]*(m[:flow_qij][l])^2 +
                    0.5*m[:mu]*(m[:flow_pji][l])^2 +
                    0.5*m[:mu]*(m[:flow_qji][l])^2 +
                    0.5*m[:mu]*(m[:rect][l])^2 +
                    0.5*m[:mu]*(m[:line_sij][l])^2 +
                    0.5*m[:mu]*(m[:line_sji][l])^2
                    for l=1:n)
            )
            t += @elapsed JuMP.optimize!(m)
            st = termination_status(m)
            if !(st in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED])
                @error("Augmented Lagrangian failed status: ", st)
            else
                # Compute the constraint violation
                cviol[1:n] .= value.(m[:pij]) .- (YffR[start:last].*value.(m[:wi_i_ij]) .+ YftR[start:last].*value.(m[:wRij]) .+ YftI[start:last].*value.(m[:wIij]))
                cviol[n+1:2*n] .= value.(m[:qij]) .- (-YffI[start:last].*value.(m[:wi_i_ij]) .- YftI[start:last].*value.(m[:wRij]) .+ YftR[start:last].*value.(m[:wIij]))
                cviol[2*n+1:3*n] .= value.(m[:pji]) .- (YttR[start:last].*value.(m[:wi_j_ji]) .+ YtfR[start:last].*value.(m[:wRij]) .- YtfI[start:last].*value.(m[:wIij]))
                cviol[3*n+1:4*n] .= value.(m[:qji]) .- (-YttI[start:last].*value.(m[:wi_j_ji]) .- YtfI[start:last].*value.(m[:wRij]) .- YtfR[start:last].*value.(m[:wIij]))
                cviol[4*n+1:5*n] .= value.(m[:wRij]).^2 .+ value.(m[:wIij]).^2 .- value.(m[:wi_i_ij]).*value.(m[:wi_j_ji])
                if param.linelimit && !isempty(ll_idx)
                    for l=1:length(ll_idx)
                        cviol[5*n+l] = value.(m[:pij][ll_idx[l]]).^2 .+ value.(m[:qij][ll_idx[l]]).^2 .- (lines[start+ll_idx[l]-1].rateA./baseMVA).^2 .+ value.(m[:sij][l])
                        cviol[6*n+l] = value.(m[:pji][ll_idx[l]]).^2 .+ value.(m[:qji][ll_idx[l]]).^2 .- (lines[start+ll_idx[l]-1].rateA./baseMVA).^2 .+ value.(m[:sji][l])
                    end
                end

                if (cnorm = norm(cviol, Inf)) <= eta
                    if cnorm <= 1e-5
                        terminate = true
                    else
                        m[:l_flow_pij] .+= m[:mu].*cviol[1:n]
                        m[:l_flow_qij] .+= m[:mu].*cviol[n+1:2*n]
                        m[:l_flow_pji] .+= m[:mu].*cviol[2*n+1:3*n]
                        m[:l_flow_qji] .+= m[:mu].*cviol[3*n+1:4*n]
                        m[:l_rect] .+= m[:mu].*cviol[4*n+1:5*n]
                        if param.linelimit && !isempty(ll_idx)
                            m[:l_line_sij][1:length(ll_idx)] .+= m[:mu].*cviol[5*n+1:5*n+length(ll_idx)]
                            m[:l_line_sji][1:length(ll_idx)] .+= m[:mu].*cviol[6*n+1:6*n+length(ll_idx)]
                        end
                        eta = eta / m[:mu]^0.9
                        omega = omega / m[:mu]
                    end
                else
                    m[:mu] *= 10
                    eta = 1 / m[:mu]^0.1
                    omega = 1 / m[:mu]
                end

                if it == 100
                    println("Maximum iteration has reached.")
                    terminate = true
                end
            end
            auglag_it += it
        end
    else
        @NLobjective(m, Min,
                        sum(l_curr[param.pij_start+start+l-1]*m[:pij][l] +
                            l_curr[param.qij_start+start+l-1]*m[:qij][l] +
                            l_curr[param.wi_i_ij_start+start+l-1]*m[:wi_i_ij][l] +
                            l_curr[param.pji_start+start+l-1]*m[:pji][l] +
                            l_curr[param.qji_start+start+l-1]*m[:qji][l] +
                            l_curr[param.wi_j_ji_start+start+l-1]*m[:wi_j_ji][l] +
                            0.5*rho[param.wi_i_ij_start+start+l-1]*(m[:wi_i_ij][l] - v_curr[param.wi_i_ij_start+start+l-1])^2 +
                            0.5*rho[param.wi_j_ji_start+start+l-1]*(m[:wi_j_ji][l] - v_curr[param.wi_j_ji_start+start+l-1])^2 +
                            0.5*rho[param.pij_start+start+l-1]*(m[:pij][l] - v_curr[param.pij_start+start+l-1])^2 +
                            0.5*rho[param.qij_start+start+l-1]*(m[:qij][l] - v_curr[param.qij_start+start+l-1])^2 +
                            0.5*rho[param.pji_start+start+l-1]*(m[:pji][l] - v_curr[param.pji_start+start+l-1])^2 +
                            0.5*rho[param.qji_start+start+l-1]*(m[:qji][l] - v_curr[param.qji_start+start+l-1])^2
                            for l=1:n)
        )

        t += @elapsed JuMP.optimize!(m)
    end

    st = termination_status(m)
    if !(st in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_OPTIMAL])
        println("Branch does not solve to optimality: ", st)
        if st in [MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE]
            info.br_infeas += 1
        elseif st == MOI.NUMERICAL_ERROR
            info.br_refail += 1
        else
            info.br_other += 1
        end
    else
        u_curr[param.pij_start+start:param.pij_start+last] .= value.(m[:pij])
        u_curr[param.qij_start+start:param.qij_start+last] .= value.(m[:qij])
        u_curr[param.pji_start+start:param.pji_start+last] .= value.(m[:pji])
        u_curr[param.qji_start+start:param.qji_start+last] .= value.(m[:qji])
        u_curr[param.wi_i_ij_start+start:param.wi_i_ij_start+last] .= value.(m[:wi_i_ij])
        u_curr[param.wi_j_ji_start+start:param.wi_j_ji_start+last] .= value.(m[:wi_j_ji])
    end

    if start <= last
        info.avg_br = t
        info.avg_auglag = auglag_it
        info.cum_avg_br += info.avg_br
    end

    return param, info
end

function solve_branches(param::AdmmParamRect, info::AdmmInfo)
    if param.use_whole
        return solve_whole_branches(param, info)
    end

    baseMVA = data.baseMVA
    buses = data.buses
    lines = data.lines
    busref = data.bus_ref
    nline = length(data.lines)
    BusIdx = data.BusIdx

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    p = myid()
    nele = floor(Int, nline / nworkers())
    nrem = mod(nline, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    u_curr = param.u_curr
    v_curr = param.v_curr
    l_curr = param.l_curr
    rho = param.rho

    t = 0.0
    auglag_it = 0
    info.br_infeas = info.br_refail = info.br_other = 0
    for l=start:last
        fr = BusIdx[lines[l].from]
        to = BusIdx[lines[l].to]

        m = brmod[l - start + 1]
        set_start_value(m[:pij], u_curr[param.pij_start+l])
        set_start_value(m[:qij], u_curr[param.qij_start+l])
        set_start_value(m[:wi_i_ij], u_curr[param.wi_i_ij_start+l])
        set_start_value(m[:pji], u_curr[param.pji_start+l])
        set_start_value(m[:qji], u_curr[param.qji_start+l])
        set_start_value(m[:wi_j_ji], u_curr[param.wi_j_ji_start+l])
        if info.it > 1
            set_start_value(m[:wRij], value(m[:wRij]))
            set_start_value(m[:wIij], value(m[:wIij]))
        end

        if param.use_auglag
            if param.linelimit && (lines[l].rateA > 0 && lines[l].rateA < 1e10)
                set_start_value(m[:sij], (lines[l].rateA/baseMVA)^2 - start_value(m[:pij])^2 - start_value(m[:qij])^2)
                set_start_value(m[:sji], (lines[l].rateA/baseMVA)^2 - start_value(m[:pji])^2 - start_value(m[:qji])^2)
            end

            it = 0
            terminate = false
            omega = 1 / m[:mu]
            eta = 1 / (m[:mu])^0.1
            cviol = zeros(7)
            scale_factor = 1e5
            mu_max = 1e8

            while !terminate
                it += 1
                @NLobjective(m, Min,
                    (1/scale_factor)*(l_curr[param.pij_start+l]*m[:pij] +
                    l_curr[param.qij_start+l]*m[:qij] +
                    l_curr[param.wi_i_ij_start+l]*m[:wi_i_ij] +
                    l_curr[param.pji_start+l]*m[:pji] +
                    l_curr[param.qji_start+l]*m[:qji] +
                    l_curr[param.wi_j_ji_start+l]*m[:wi_j_ji] +
                    0.5*rho[param.wi_i_ij_start+l]*(m[:wi_i_ij] - v_curr[param.wi_i_ij_start+l])^2 +
                    0.5*rho[param.wi_j_ji_start+l]*(m[:wi_j_ji] - v_curr[param.wi_j_ji_start+l])^2 +
                    0.5*rho[param.pij_start+l]*(m[:pij] - v_curr[param.pij_start+l])^2 +
                    0.5*rho[param.qij_start+l]*(m[:qij] - v_curr[param.qij_start+l])^2 +
                    0.5*rho[param.pji_start+l]*(m[:pji] - v_curr[param.pji_start+l])^2 +
                    0.5*rho[param.qji_start+l]*(m[:qji] - v_curr[param.qji_start+l])^2 +
                    m[:l_flow_pij]*m[:flow_pij] +
                    m[:l_flow_qij]*m[:flow_qij] +
                    m[:l_flow_pji]*m[:flow_pji] +
                    m[:l_flow_qji]*m[:flow_qji] +
                    m[:l_rect]*m[:rect] +
                    m[:l_line_sij]*m[:line_sij] +
                    m[:l_line_sji]*m[:line_sji] +
                    0.5*m[:mu]*(m[:flow_pij])^2 +
                    0.5*m[:mu]*(m[:flow_qij])^2 +
                    0.5*m[:mu]*(m[:flow_pji])^2 +
                    0.5*m[:mu]*(m[:flow_qji])^2 +
                    0.5*m[:mu]*(m[:rect])^2 +
                    0.5*m[:mu]*(m[:line_sij])^2 +
                    0.5*m[:mu]*(m[:line_sji])^2)
                )
                t += @elapsed JuMP.optimize!(m)
                st = termination_status(m)
                if st != MOI.LOCALLY_SOLVED
                    println("Augmented Lagrangian failed for branch ", l, " with status ", st)
                    println("Resolve with proximal perturbation . . .")
                    pert = 0.1
                    pert_it = 0
                    pij_v = value(m[:pij])
                    qij_v = value(m[:qij])
                    pji_v = value(m[:pji])
                    qji_v = value(m[:qji])
                    wi_i_ij_v = value(m[:wi_i_ij])
                    wi_j_ji_v = value(m[:wi_j_ji])
                    wRij_v = value(m[:wRij])
                    wIij_v = value(m[:wIij])
                    while (pert_it < 50 && pert >= 1e-6)
                        @NLobjective(m, Min,
                        0.5*pert*((m[:pij] - pij_v)^2 +
                                  (m[:qij] - qij_v)^2 +
                                  (m[:pji] - pji_v)^2 +
                                  (m[:qji] - qji_v)^2 +
                                  (m[:wi_i_ij] - wi_i_ij_v)^2 +
                                  (m[:wi_j_ji] - wi_j_ji_v)^2 +
                                  (m[:wRij] - wRij_v)^2 +
                                  (m[:wIij] - wIij_v)^2
                                ) +
                        (1/scale_factor)*(l_curr[param.pij_start+l]*m[:pij] +
                        l_curr[param.qij_start+l]*m[:qij] +
                        l_curr[param.wi_i_ij_start+l]*m[:wi_i_ij] +
                        l_curr[param.pji_start+l]*m[:pji] +
                        l_curr[param.qji_start+l]*m[:qji] +
                        l_curr[param.wi_j_ji_start+l]*m[:wi_j_ji] +
                        0.5*rho[param.wi_i_ij_start+l]*(m[:wi_i_ij] - v_curr[param.wi_i_ij_start+l])^2 +
                        0.5*rho[param.wi_j_ji_start+l]*(m[:wi_j_ji] - v_curr[param.wi_j_ji_start+l])^2 +
                        0.5*rho[param.pij_start+l]*(m[:pij] - v_curr[param.pij_start+l])^2 +
                        0.5*rho[param.qij_start+l]*(m[:qij] - v_curr[param.qij_start+l])^2 +
                        0.5*rho[param.pji_start+l]*(m[:pji] - v_curr[param.pji_start+l])^2 +
                        0.5*rho[param.qji_start+l]*(m[:qji] - v_curr[param.qji_start+l])^2 +
                        m[:l_flow_pij]*m[:flow_pij] +
                        m[:l_flow_qij]*m[:flow_qij] +
                        m[:l_flow_pji]*m[:flow_pji] +
                        m[:l_flow_qji]*m[:flow_qji] +
                        m[:l_rect]*m[:rect] +
                        m[:l_line_sij]*m[:line_sij] +
                        m[:l_line_sji]*m[:line_sji] +
                        0.5*m[:mu]*(m[:flow_pij])^2 +
                        0.5*m[:mu]*(m[:flow_qij])^2 +
                        0.5*m[:mu]*(m[:flow_pji])^2 +
                        0.5*m[:mu]*(m[:flow_qji])^2 +
                        0.5*m[:mu]*(m[:rect])^2 +
                        0.5*m[:mu]*(m[:line_sij])^2 +
                        0.5*m[:mu]*(m[:line_sji])^2)
                        )
                        JuMP.optimize!(m)
                        st = termination_status(m)
                        println("Resolve status ", termination_status(m))
                        if st != MOI.LOCALLY_SOLVED
                            pert = pert * 10.0
                        else
                            pert = pert / 10.0
                        end

                        pert_it = pert_it + 1

                        set_start_value(m[:pij], value(m[:pij]))
                        set_start_value(m[:qij], value(m[:qij]))
                        set_start_value(m[:wi_i_ij], value(m[:wi_i_ij]))
                        set_start_value(m[:pji], value(m[:pji]))
                        set_start_value(m[:qji], value(m[:qji]))
                        set_start_value(m[:wi_j_ji], value(m[:wi_j_ji]))
                        set_start_value(m[:wRij], value(m[:wRij]))
                        set_start_value(m[:wIij], value(m[:wIij]))
                        pij_v = value(m[:pij])
                        qij_v = value(m[:qij])
                        pji_v = value(m[:pji])
                        qji_v = value(m[:qji])
                        wi_i_ij_v = value(m[:wi_i_ij])
                                wi_j_ji_v = value(m[:wi_j_ji])
                        wRij_v = value(m[:wRij])
                        wIij_v = value(m[:wIij])
                    end
                    if pert > 1e-6
                        error("Augmented Lagrangian failed for branch ", l, " status: ", st)
                    end
                end

                if st == MOI.LOCALLY_SOLVED
                    # Compute the constraint violation
                    cviol[1] = value.(m[:pij]) - (YffR[l]*value.(m[:wi_i_ij]) + YftR[l]*value.(m[:wRij]) + YftI[l]*value.(m[:wIij]))
                    cviol[2] = value.(m[:qij]) - (-YffI[l]*value.(m[:wi_i_ij]) - YftI[l]*value.(m[:wRij]) + YftR[l]*value.(m[:wIij]))
                    cviol[3] = value.(m[:pji]) - (YttR[l]*value.(m[:wi_j_ji]) + YtfR[l]*value.(m[:wRij]) - YtfI[l]*value.(m[:wIij]))
                    cviol[4] = value.(m[:qji]) - (-YttI[l]*value.(m[:wi_j_ji]) - YtfI[l]*value.(m[:wRij]) - YtfR[l]*value.(m[:wIij]))
                    cviol[5] = value.(m[:wRij]).^2 .+ value.(m[:wIij]).^2 .- value.(m[:wi_i_ij]).*value.(m[:wi_j_ji])
                    if param.linelimit && (lines[l].rateA > 0 && lines[l].rateA < 1e10)
                        cviol[6] = value.(m[:pij]).^2 .+ value.(m[:qij]).^2 .- (lines[l].rateA./baseMVA).^2 .+ value.(m[:sij])
                        cviol[7] = value.(m[:pji]).^2 .+ value.(m[:qji]).^2 .- (lines[l].rateA./baseMVA).^2 .+ value.(m[:sji])
                    end

                    cnorm = norm(cviol, Inf)
                    if cnorm <= eta
                        if cnorm <= 1e-6
                            terminate = true
                        else
                            m[:l_flow_pij] += m[:mu]*cviol[1]
                            m[:l_flow_qij] += m[:mu]*cviol[2]
                            m[:l_flow_pji] += m[:mu]*cviol[3]
                            m[:l_flow_qji] += m[:mu]*cviol[4]
                            m[:l_rect] += m[:mu]*cviol[5]
                            m[:l_line_sij] += m[:mu]*cviol[6]
                            m[:l_line_sji] += m[:mu]*cviol[7]
                            eta = eta / m[:mu]^0.9
                            omega = omega / m[:mu]
                        end
                    else
                        m[:mu] = min(mu_max, m[:mu]*10)
                        eta = 1 / m[:mu]^0.1
                        omega = 1 / m[:mu]
                    end

                    if (it % 100) == 0
                        mu_max = min(1e12, mu_max*10.0)
                        if it == 500
                            println("Branch ", l, " maximum iteration has reached.")
                            terminate = true
                        end
                    end
                end
            end
            auglag_it += it
        else
            @NLobjective(m, Min,
                            l_curr[param.pij_start+l]*m[:pij] +
                            l_curr[param.qij_start+l]*m[:qij] +
                            l_curr[param.wi_i_ij_start+l]*m[:wi_i_ij] +
                            l_curr[param.pji_start+l]*m[:pji] +
                            l_curr[param.qji_start+l]*m[:qji] +
                            l_curr[param.wi_j_ji_start+l]*m[:wi_j_ji] +
                            0.5*rho[param.wi_i_ij_start+l]*(m[:wi_i_ij] - v_curr[param.wi_i_ij_start+l])^2 +
                            0.5*rho[param.wi_j_ji_start+l]*(m[:wi_j_ji] - v_curr[param.wi_j_ji_start+l])^2 +
                            0.5*rho[param.pij_start+l]*(m[:pij] - v_curr[param.pij_start+l])^2 +
                            0.5*rho[param.qij_start+l]*(m[:qij] - v_curr[param.qij_start+l])^2 +
                            0.5*rho[param.pji_start+l]*(m[:pji] - v_curr[param.pji_start+l])^2 +
                            0.5*rho[param.qji_start+l]*(m[:qji] - v_curr[param.qji_start+l])^2
            )

            t += @elapsed JuMP.optimize!(m)
        end
        st = termination_status(m)
        if !(st in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_OPTIMAL])
            println("Branch ", l, " does not solve to optimality: ", st)
            if st in [MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE]
                info.br_infeas += 1
            elseif st == MOI.NUMERICAL_ERROR
                info.br_refail += 1
            else
                info.br_other += 1
            end
        else
            u_curr[param.pij_start+l] = value.(m[:pij])
            u_curr[param.qij_start+l] = value.(m[:qij])
            u_curr[param.pji_start+l] = value.(m[:pji])
            u_curr[param.qji_start+l] = value.(m[:qji])
            u_curr[param.wi_i_ij_start+l] = value.(m[:wi_i_ij])
            u_curr[param.wi_j_ji_start+l] = value.(m[:wi_j_ji])
        end
    end

    if start <= last
        info.avg_br = t / (last-start+1)
        info.avg_auglag = auglag_it / (last-start+1)
        info.cum_avg_br += info.avg_br
    end

    return param, info
end

function solve_buses(param::AdmmParamRect, info::AdmmInfo)
    baseMVA = data.baseMVA
    buses = data.buses
    gens = data.generators
    busref = data.bus_ref
    nbus = length(data.buses)
    BusGens = data.BusGenerators
    FromLines = data.FromLines
    ToLines = data.ToLines

    YshR = ybus.YshR; YshI = ybus.YshI

    p = myid()
    nele = floor(Int, nbus / nworkers())
    nrem = mod(nbus, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    u_curr = param.u_curr
    v_curr = param.v_curr
    l_curr = param.l_curr
    rho = param.rho

    A = zeros(2,2)
    t = 0
    for b=start:last
        # A closed-form solution exists. Using dL/dx = 0, we can rewrite all
        # primal variables in terms of dual variables mu_p and mu_q as below:
        #
        # dL/dpg_i = -lam_pg - rho_pq*(pg - pg_i) + mu_p = 0
        #        ==> pg_i = pg + ( (lam_pg - mu_p) / rho_pq )
        # dL/dqg_i = -lam_qg - rho_pq*(qg - qg_i) + mu_q = 0
        #        ==> qg_i = qg + ( (lam_qg - mu_q) / rho_pq )
        # dL/dpij  = -lam_pij - rho_pq*(pij - pij_i) - mu_p = 0
        #        ==> pij_i = pij + ( (lam_pij + mu_p) / rho_pq )
        # dL/dqij  = -lam_qij - rho_pq*(qij - qij_i) - mu_q = 0
        #        ==> qij_i = qij + ( (lam_qij + mu_q) / rho_pq )
        # dL/dpji  = -lam_pji - rho_pq*(pji - pji_i) - mu_p = 0
        #        ==> pji_i = pji + ( (lam_pji + mu_p) / rho_pq )
        # dL/dqji  = -lam_qji - rho_pq*(qji - qji_i) - mu_q = 0
        #        ==> qji_i = qji + ( (lam_qji + mu_q) / rho_pq )
        # dL/dwi   = sum_{ij} ( -lam_wi_i_ij - rho_va*(wi_i_ij - wi) ) +
        #            sum_{ji} ( -lam_wi_j_ji - rho_va*(wi_j_ji - wi) )
        #            - YshR*mu_p + YshI*mu_q = 0
        #        ==> wi = [ sum_{ij} ( lam_wi_i_ij + rho_va*wi_i_ij ) +
        #                   sum_{ji} ( lam_wi_j_ji + rho_va*wi_j_ji )
        #                   + YshR*mu_p - YshI*mu_q ] / [ rho_va*(sum_{ij} 1 + sum)_{ji} 1) ]
        #
        # By plug-in these primal variables to the power balance equation,
        # we compute mu_p and mu_q:
        #
        #   sum_gi pg_i - pd_i - [ sum_{ij} pij_i + sum_{ji} pji_i + YshR*wi ] = 0
        #   sum_qi qg_i - qd_i - [ sum_{ij} qij_i + sum_{ji} qji_i - YshI*wi ] = 0
        #
        # Then we end up with the following system of equations:
        #
        #   / A_11  A_12 \ / mu_p \  = / rhs1 \
        #   \ A_21  A_22 / \ mu_q /    \ rhs2 /
        #
        # where
        #  A_11 = (sum_g + sum_{ij} + sum_{ji})/rho_pq + YshR^2/(rho_va*(sum_{ij}+sum_{ji}))
        #  A_12 = (-YshR*(YshI / (rho_va*(sum_{ij}+sum_{ji}))))
        #  A_21 = (-YshI*(YshR / (rho_va*(sum_{ij}+sum_{ji}))))
        #  A_22 = (sum_g + sum_{ij} + sum_{ji})/rho_pq + YshI^2/(rho_va*(sum_{ij}+sum_{ji}))
        #  rhs1 = (sum_g pg + (lam_pg/rho_pq) + (-pd_i/baseMVA)
        #          - [ sum_{ij} ( pij + (lam_pij/rho_pq) ) +
        #              sum_{ji} ( pji + (lam_pji/rho_pq) ) +
        #              YshR*{ ( sum_{ij} lam_wi_i_ij + rho_va*wi_i_ij +
        #                       sum_{ji} lam_wi_j_ji + rho_va*wi_j_ji ) / (rho_va*(sum_{ij}+sum_{ji}))
        #                      )
        #                   }
        #            ]
        #  rhs2 = (sum_g qg + (lam_qg/rho_pq) + (-qd_i/baseMVA)
        #          - [ sum_{ij} ( qij + (lam_qij/rho_pq) ) +
        #              sum_{ji} ( qji + (lam_qji/rho_pq) ) -
        #              YshI*{ ( sum_{ij} lam_wi_i_ij + rho_va*wi_i_ij +
        #                       sum_{ji} lam_wi_j_ji + rho_va*wi_j_ji ) / (rho_va*(sum_{ij}+sum_{ji}))
        #                      )
        #                   }
        #            ]

        t += @elapsed begin
            common = 0
            inv_rhosum_pij_ji = 0
            inv_rhosum_qij_ji = 0
            rhosum_wi_ij_ji = 0
            if !isempty(FromLines[b])
                common += sum(l_curr[param.wi_i_ij_start+l] + rho[param.wi_i_ij_start+l]*u_curr[param.wi_i_ij_start+l] for l in FromLines[b])
                inv_rhosum_pij_ji += sum(1.0 / rho[param.pij_start+l] for l in FromLines[b])
                inv_rhosum_qij_ji += sum(1.0 / rho[param.qij_start+l] for l in FromLines[b])
                rhosum_wi_ij_ji += sum(rho[param.wi_i_ij_start+l] for l in FromLines[b])
            end
            if !isempty(ToLines[b])
                common += sum(l_curr[param.wi_j_ji_start+l] + rho[param.wi_j_ji_start+l]*u_curr[param.wi_j_ji_start+l] for l in ToLines[b])
                inv_rhosum_pij_ji += sum(1.0 / rho[param.pji_start+l] for l in ToLines[b])
                inv_rhosum_qij_ji += sum(1.0 / rho[param.qji_start+l] for l in ToLines[b])
                rhosum_wi_ij_ji += sum(rho[param.wi_j_ji_start+l] for l in ToLines[b])
            end
            common /= rhosum_wi_ij_ji

            rhs1 = 0
            rhs2 = 0
            inv_rhosum_pg = 0
            inv_rhosum_qg = 0
            if !isempty(BusGens[b])
                rhs1 += sum(u_curr[param.pg_start+g] + (l_curr[param.pg_start+g]/rho[param.pg_start+g]) for g in BusGens[b])
                rhs2 += sum(u_curr[param.qg_start+g] + (l_curr[param.qg_start+g]/rho[param.qg_start+g]) for g in BusGens[b])
                inv_rhosum_pg += sum(1.0 / rho[param.pg_start+g] for g in BusGens[b])
                inv_rhosum_qg += sum(1.0 / rho[param.qg_start+g] for g in BusGens[b])
            end
            rhs1 -= (buses[b].Pd / baseMVA)
            rhs2 -= (buses[b].Qd / baseMVA)

            if !isempty(FromLines[b])
                rhs1 -= sum(u_curr[param.pij_start+l] + (l_curr[param.pij_start+l]/rho[param.pij_start+l]) for l in FromLines[b])
                rhs2 -= sum(u_curr[param.qij_start+l] + (l_curr[param.qij_start+l]/rho[param.qij_start+l]) for l in FromLines[b])
            end
            if !isempty(ToLines[b])
                rhs1 -= sum(u_curr[param.pji_start+l] + (l_curr[param.pji_start+l]/rho[param.pji_start+l]) for l in ToLines[b])
                rhs2 -= sum(u_curr[param.qji_start+l] + (l_curr[param.qji_start+l]/rho[param.qji_start+l]) for l in ToLines[b])
            end
            rhs1 -= YshR[b]*common
            rhs2 += YshI[b]*common

            A[1,1] = (inv_rhosum_pg + inv_rhosum_pij_ji) + (YshR[b]^2 / rhosum_wi_ij_ji)
            A[1,2] = -YshR[b]*(YshI[b] / rhosum_wi_ij_ji)
            A[2,1] = A[1,2]
            A[2,2] = (inv_rhosum_qg + inv_rhosum_qij_ji) + (YshI[b]^2 / rhosum_wi_ij_ji)
            mu = A \ [rhs1 ; rhs2]
            wi = common + ( (YshR[b]*mu[1] - YshI[b]*mu[2]) / rhosum_wi_ij_ji )

            for g in BusGens[b]
                v_curr[param.pg_start+g] = u_curr[param.pg_start+g] + (l_curr[param.pg_start+g] - mu[1]) / rho[param.pg_start+g]
                v_curr[param.qg_start+g] = u_curr[param.qg_start+g] + (l_curr[param.qg_start+g] - mu[2]) / rho[param.qg_start+g]
            end
            for l in FromLines[b]
                v_curr[param.pij_start+l] = u_curr[param.pij_start+l] + (l_curr[param.pij_start+l] + mu[1]) / rho[param.pij_start+l]
                v_curr[param.qij_start+l] = u_curr[param.qij_start+l] + (l_curr[param.qij_start+l] + mu[2]) / rho[param.qij_start+l]
                v_curr[param.wi_i_ij_start+l] = wi
            end
            for l in ToLines[b]
                v_curr[param.pji_start+l] = u_curr[param.pji_start+l] + (l_curr[param.pji_start+l] + mu[1]) / rho[param.pji_start+l]
                v_curr[param.qji_start+l] = u_curr[param.qji_start+l] + (l_curr[param.qji_start+l] + mu[2]) / rho[param.qji_start+l]
                v_curr[param.wi_j_ji_start+l] = wi
            end
        end
    end

    if start <= last
        info.avg_bus = t / (last-start+1)
        info.cum_avg_bus += info.avg_bus
    end

    return param, info
end

function init_param(data::OPFData, rho_pq::Float64, rho_va::Float64)
    gens = data.generators
    nbus = length(data.buses)
    ngen = length(data.generators)
    nline = length(data.lines)
    
    param = AdmmParamRect(ngen, nline, nbus, rho_pq, rho_va)
    info = AdmmInfo()

    for g=1:ngen
        param.v_curr[param.pg_start+g] = 0.5*(gens[g].Pmin + gens[g].Pmax)
        param.v_curr[param.qg_start+g] = 0.5*(gens[g].Qmin + gens[g].Qmax)
    end

    for l=1:nline
        param.v_curr[param.wi_i_ij_start+l] = 1.0
        param.v_curr[param.wi_j_ji_start+l] = 1.0
    end

    return param, info
end

function load_solution(solfile::AbstractString, param::AdmmParamRect, info::AdmmInfo)
    sol = load(solfile)   # Dict{String,Any}
    @assert(param.ngen == sol["ngen"])
    @assert(param.nline == sol["nline"])
    @assert(param.nbus == sol["nbus"])

    nvars = 2*param.ngen + 6*param.nline

    info.it = sol["it"]
    info.gmax = sol["gmax"]
    info.obj = sol["obj"]
    info.primres = sol["primres"]
    info.dualres = sol["dualres"]

    copyto!(param.rho, 1, sol["rho"], 1, nvars)
    copyto!(param.u_curr, 1, sol["u_curr"], 1, nvars)
    copyto!(param.u_prev, 1, sol["u_prev"], 1, nvars)
    copyto!(param.v_curr, 1, sol["v_curr"], 1, nvars)
    copyto!(param.v_prev, 1, sol["v_prev"], 1, nvars)
    copyto!(param.l_curr, 1, sol["l_curr"], 1, nvars)
    copyto!(param.l_prev, 1, sol["l_prev"], 1, nvars)
end

function save_solution(solfile::AbstractString, param::AdmmParamRect, info::AdmmInfo)
    save(solfile,
         "it", info.it, "gmax", info.gmax, "obj", info.obj,
         "primres", info.primres, "dualres", info.dualres,
         "ngen", param.ngen, "nline", param.nline, "nbus", param.nbus,
         "rho", param.rho, "u_curr", param.u_curr, "u_prev", param.u_prev,
         "v_curr", param.v_curr, "v_prev", param.v_prev,
         "l_curr", param.l_curr, "l_prev", param.l_prev)
end

function print_iteration_title()
    @printf("%10s  %12s  %12s  %12s  %12s  %12s  %12s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %12s\n",
            "Iterations", "PriRes", "DualRes", "Objective", "||g||",
            "EpsPri", "EpsDual", "RhoPQMax", "RhoPQMin", "RhoVAMax", "RhoVAMin",
            "BrTime", "BrFails", "BrAugLag", "Elapsed")
end

function print_iteration(info::AdmmInfo; res_io=nothing)
    if (info.it % 50) == 0
        print_iteration_title()
    end

    nfail = info.br_infeas + info.br_refail + info.br_other
    @printf("%10d  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %8.2f  %8.2f  %8.2f  %8.2f  %8.4f  %8d  %8.4f  %12.4f\n",
            info.it, info.primres, info.dualres, info.obj, info.gmax,
            info.eps_pri, info.eps_dual, info.rho_pq_max, info.rho_pq_min,
            info.rho_va_max, info.rho_va_min, info.avg_br, nfail, info.avg_auglag, info.elapsed)
    if res_io !== nothing
        @printf(res_io, "%10d  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %8.2f  %8.2f  %8.2f  %8.2f  %8.4f  %8d  %8.4f  %12.4f\n",
                info.it, info.primres, info.dualres, info.obj, info.gmax,
                info.eps_pri, info.eps_dual, info.rho_pq_max, info.rho_pq_min,
                info.rho_va_max, info.rho_va_min, info.avg_br, nfail, info.avg_auglag, info.elapsed)
        flush(res_io)
    end
end

function print_iteration_end_message(info::AdmmInfo; io=nothing)
    @printf("\n")
    if info.term == :LocalOptimal
        @printf(" ** A solution found.\n\n")
    elseif info.term == :IterLimit
        @printf(" ** Iteration limit.\n\n")
    end

    @printf("Iterations . . . . . . . . . . . . . % 12d\n", info.it)
    @printf("Primal residual. . . . . . . . . . . % .5e\n", info.primres)
    @printf("Dual residual. . . . . . . . . . . . % .5e\n", info.dualres)
    @printf("Objective. . . . . . . . . . . . . . % .5e\n", info.obj)
    @printf("Feasibility. . . . . . . . . . . . . % .5e\n", info.gmax)
    @printf("EpsPri . . . . . . . . . . . . . . . % .5e\n", info.eps_pri)
    @printf("EpsDual  . . . . . . . . . . . . . . % .5e\n", info.eps_dual)
    @printf("Number of branch infeasibility . . . % 12d\n", info.br_infeas)
    @printf("Number of branch restoration fail. . % 12d\n", info.br_refail)
    @printf("Number of branch other errors. . . . % 12d\n", info.br_other)
    @printf("Elapsed time (secs). . . . . . . . . % 12.3f\n", info.elapsed)
    @printf("Average parallel generators (secs) . % 12.3f\n", info.cum_avg_gen / info.it)
    @printf("Average parallel branches (secs) . . % 12.3f\n", info.cum_avg_br / info.it)
    @printf("Average parallel buses (secs)  . . . % 12.3f\n", info.cum_avg_bus / info.it)

    if io !== nothing
        @printf(io, "\n")
        if info.term == :LocalOptimal
            @printf(io, " ** A solution found.\n\n")
        elseif info.term == :IterLimit
            @printf(io, " ** Iteration limit.\n\n")
        end

        @printf(io, "Iterations . . . . . . . . . . . . . % 12d\n", info.it)
        @printf(io, "Primal residual. . . . . . . . . . . % .5e\n", info.primres)
        @printf(io, "Dual residual. . . . . . . . . . . . % .5e\n", info.dualres)
        @printf(io, "Objective. . . . . . . . . . . . . . % .5e\n", info.obj)
        @printf(io, "Feasibility. . . . . . . . . . . . . % .5e\n", info.gmax)
        @printf(io, "EpsPri . . . . . . . . . . . . . . . % .5e\n", info.eps_pri)
        @printf(io, "EpsDual  . . . . . . . . . . . . . . % .5e\n", info.eps_dual)
        @printf(io, "Number of branch infeasibility . . . % 12d\n", info.br_infeas)
        @printf(io, "Number of branch restoration fail. . % 12d\n", info.br_refail)
        @printf(io, "Number of branch other errors. . . . % 12d\n", info.br_other)
        @printf(io, "Elapsed time (secs). . . . . . . . . % 12.3f\n", info.elapsed)
        @printf(io, "Average parallel generators (secs) . % 12.3f\n", info.cum_avg_gen / info.it)
        @printf(io, "Average parallel branches (secs) . . % 12.3f\n", info.cum_avg_br / info.it)
        @printf(io, "Average parallel buses (secs)  . . . % 12.3f\n", info.cum_avg_bus / info.it)
    end
end

function get_new_filename(prefix::AbstractString, ext::AbstractString)
    outfile = prefix
    if isfile(outfile*ext)
        num = 1
        numcopy = @sprintf("_%d", num)
        while isfile(outfile*numcopy*ext)
            num += 1
            numcopy = @sprintf("_%d", num)
        end
        outfile = outfile*numcopy
    end

    return outfile
end

function parallel_solve_generators(fts::Vector{Future}, param::AdmmParamRect, info::AdmmInfo)
    # Solve generators in parallel in a distributed fashion.
    for p in workers()
        fts[p] = remotecall(solve_generators, p, param, info)
    end

    # Collect the results.
    ngen = length(data.generators)
    nele = floor(Int, ngen / nworkers())
    nrem = mod(ngen, nworkers())
    avg_gen = 0

    u_curr = param.u_curr

    for p in workers()
        w_param, w_info = fetch(fts[p])
        start = get_start(p, nele, nrem)
        last = start + get_inc(p, nele, nrem)
        if start <= last
            for g=start:last
                u_curr[param.pg_start+g] = w_param.u_curr[param.pg_start+g]
                u_curr[param.qg_start+g] = w_param.u_curr[param.qg_start+g]
            end
            avg_gen += w_info.avg_gen * (last-start+1)
        end
    end
    info.avg_gen = avg_gen / ngen
    info.cum_avg_gen += info.avg_gen
end

function parallel_solve_branches(fts::Vector{Future}, param::AdmmParamRect, info::AdmmInfo)
    # Solve branches in parallel in a distributed fashion.
    for p in workers()
        fts[p] = remotecall(solve_branches, p, param, info)
    end

    # Collect the results.
    nline = length(data.lines)
    nele = floor(Int, nline / nworkers())
    nrem = mod(nline, nworkers())
    avg_br = 0
    avg_auglag = 0

    u_curr = param.u_curr

    for p in workers()
        w_param, w_info = fetch(fts[p])
        start = get_start(p, nele, nrem)
        last = start + get_inc(p, nele, nrem)
        if start <= last
            for l=start:last
                u_curr[param.pij_start+l] = w_param.u_curr[param.pij_start+l]
                u_curr[param.qij_start+l] = w_param.u_curr[param.qij_start+l]
                u_curr[param.pji_start+l] = w_param.u_curr[param.pji_start+l]
                u_curr[param.qji_start+l] = w_param.u_curr[param.qji_start+l]
                u_curr[param.wi_i_ij_start+l] = w_param.u_curr[param.wi_i_ij_start+l]
                u_curr[param.wi_j_ji_start+l] = w_param.u_curr[param.wi_j_ji_start+l]
            end
            avg_br += w_info.avg_br * (last-start+1)
            avg_auglag += w_info.avg_auglag * (last-start+1)
            info.br_infeas += w_info.br_infeas
            info.br_refail += w_info.br_refail
            info.br_other += w_info.br_other
        end
    end
    info.avg_br = avg_br / nline
    info.avg_auglag = avg_auglag / nline
    info.cum_avg_br += info.avg_br
end

function parallel_solve_buses(fts::Vector{Future}, param::AdmmParamRect, info::AdmmInfo)
    # Solve buses in parallel in a distributed fashion.
    for p in workers()
        fts[p] = remotecall(solve_buses, p, param, info)
    end

    # Collect the results.
    BusGens = data.BusGenerators
    FromLines = data.FromLines
    ToLines = data.ToLines

    nbus = length(data.buses)
    nele = floor(Int, nbus / nworkers())
    nrem = mod(nbus, nworkers())
    avg_bus = 0

    v_curr = param.v_curr

    for p in workers()
        w_param, w_info = fetch(fts[p])
        start = get_start(p, nele, nrem)
        last = start + get_inc(p, nele, nrem)
        if start <= last
            for b=start:last
                for g in BusGens[b]
                    v_curr[param.pg_start+g] = w_param.v_curr[param.pg_start+g]
                    v_curr[param.qg_start+g] = w_param.v_curr[param.qg_start+g]
                end
                for l in FromLines[b]
                    v_curr[param.pij_start+l] = w_param.v_curr[param.pij_start+l]
                    v_curr[param.qij_start+l] = w_param.v_curr[param.qij_start+l]
                    v_curr[param.wi_i_ij_start+l] = w_param.v_curr[param.wi_i_ij_start+l]
                end
                for l in ToLines[b]
                    v_curr[param.pji_start+l] = w_param.v_curr[param.pji_start+l]
                    v_curr[param.qji_start+l] = w_param.v_curr[param.qji_start+l]
                    v_curr[param.wi_j_ji_start+l] = w_param.v_curr[param.wi_j_ji_start+l]
                end
            end
            avg_bus += w_info.avg_bus * (last-start+1)
            info.dualres = max(info.dualres, w_info.dualres)
        end
    end
    info.avg_bus = avg_bus / nbus
    info.cum_avg_bus += info.avg_bus
end

function get_objval(param::AdmmParamRect, u::Vector{Float64})::Float64
    baseMVA = data.baseMVA
    gens = data.generators
    ngen = length(data.generators)

    return ( sum(gens[g].coeff[gens[g].n-2]*(baseMVA*u[param.pg_start+g])^2 +
               gens[g].coeff[gens[g].n-1]*(baseMVA*u[param.pg_start+g]) +
               gens[g].coeff[gens[g].n]
               for g in 1:ngen) )
end

function get_gmax(param::AdmmParamRect, u::Vector{Float64}, v::Vector{Float64})::Float64
    baseMVA = data.baseMVA
    buses = data.buses
    lines = data.lines
    BusIdx = data.BusIdx
    BusGens = data.BusGenerators
    FromLines = data.FromLines
    ToLines = data.ToLines
    nbus = length(data.buses)
    nline = length(data.lines)

    gmax = 0
    for b=1:nbus
        pbal = 0
        qbal = 0

        for g in BusGens[b]
            pbal += baseMVA*u[param.pg_start+g]
            qbal += baseMVA*u[param.qg_start+g]
        end

        pbal -= buses[b].Pd
        qbal -= buses[b].Qd
        pbal /= baseMVA
        qbal /= baseMVA

        wi = 0
        for l in FromLines[b]
            pbal -= u[param.pij_start+l]
            qbal -= u[param.qij_start+l]
            wi = v[param.wi_i_ij_start+l]
        end

        for l in ToLines[b]
            pbal -= u[param.pji_start+l]
            qbal -= u[param.qji_start+l]
            wi = v[param.wi_j_ji_start+l]
        end

        pbal -= ybus.YshR[b]*wi
        qbal += ybus.YshI[b]*wi
        gmax = max(gmax, abs(pbal), abs(qbal))
    end

    return gmax
end

function adjust_rho(param::AdmmParamRect, info::AdmmInfo, tau::Array{Float64,2})
    param.delta_u .= param.u_curr .- param.u_prev
    param.delta_v .= param.v_curr .- param.v_prev
    param.delta_l .= param.l_curr .- param.l_prev
    param.alpha .= abs.(param.delta_l ./ param.delta_u)
    param.beta .= abs.(param.delta_l ./ param.delta_v)

    k = info.it
    n = length(param.rho)

    for i=1:n
        if abs(param.delta_l[i]) <= param.eps_rp_min
            tau[i,k] = tau[i,k-1]
        elseif abs(param.delta_u[i]) <= param.eps_rp_min && abs(param.delta_v[i]) > param.eps_rp_min
            tau[i,k] = param.beta[i]
        elseif abs(param.delta_u[i]) > param.eps_rp_min && abs(param.delta_v[i]) <= param.eps_rp_min
            tau[i,k] = param.alpha[i]
        elseif abs(param.delta_u[i]) <= param.eps_rp_min && abs(param.delta_v[i]) <= param.eps_rp_min
            tau[i,k] = tau[i,k-1]
        else
            tau[i,k] = sqrt(param.alpha[i]*param.beta[i])
        end
    end

    if (k % param.Kf) == 0
        mean_tau = mean(tau[:,k-param.Kf_mean+1:k]; dims=2)
        for i=1:n
            if mean_tau[i] >= param.rt_inc*param.rho[i]
                if abs(param.rp[i]) >= param.eps_rp && abs(param.rp_old[i]) >= param.eps_rp
                    if abs(param.rp[i]) > param.eta*abs(param.rp_k0[i]) || abs(param.rp_old[i]) > param.eta*abs(param.rp_k0[i])
                        param.rho[i] *= param.rt_inc
                    end
                end
                #=
            elseif mean_tau[i] > param.rt_inc2*param.rho[i]
                if abs(param.rp[i]) >= param.eps_rp && abs(param.rp_old[i]) >= param.eps_rp
                    if abs(param.rp[i]) > param.eta*abs(param.rp_k0[i]) || abs(param.rp_old[i]) > param.eta*abs(param.rp_k0[i])
                        param.rho[i] = param.rt_inc2*mean_tau[i]
                    end
                end
                =#
            elseif mean_tau[i] > param.rho[i]
                if abs(param.rp[i]) >= param.eps_rp && abs(param.rp_old[i]) >= param.eps_rp
                    if abs(param.rp[i]) > param.eta*abs(param.rp_k0[i]) || abs(param.rp_old[i]) > param.eta*abs(param.rp_k0[i])
                        param.rho[i] = mean_tau[i]
                    end
                end
            elseif mean_tau[i] <= param.rho[i]/param.rt_dec
                param.rho[i] /= param.rt_dec
                #=
            elseif mean_tau[i] < param.rho[i]/param.rt_dec2
                param.rho[i] = mean_tau[i]/param.rt_dec2
                =#
            elseif mean_tau[i] < param.rho[i]
                param.rho[i] = mean_tau[i]
            end
        end
    end

    pq_end = 2*param.ngen+4*param.nline
    param.rho[param.rho .>= param.rho_max] .= param.rho_max
    param.rho[1:pq_end] .= (param.rho[1:pq_end] .>= param.rho_min_pq) .* param.rho[1:pq_end] .+ (param.rho[1:pq_end] .< param.rho_min_pq) .* param.rho_min_pq
    param.rho[pq_end+1:end] .= (param.rho[pq_end+1:end] .>= param.rho_min_w) .* param.rho[pq_end+1:end] .+ (param.rho[pq_end+1:end] .< param.rho_min_w) .* param.rho_min_w
end


function admm_rect_restart(param, info, tau; iterlim=100, increase_tau=false, log=false, update_rho=false, final_message=true, savesol=false, solfile="", outfile="", fts=nothing)
    ABSTOL = 1e-6
    RELTOL = 1e-5
    rho_pq_max = zeros(iterlim)
    rho_pq_min = zeros(iterlim)
    rho_va_max = zeros(iterlim)
    rho_va_min = zeros(iterlim)
    eps_pri = zeros(iterlim)
    eps_dual = zeros(iterlim)


    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)

    #if outfile == ""
    #    outfile = get_new_filename(param.casename*"_result_nworkers"*string(nworkers()), ".txt")
    #end
    #if solfile == ""
    #    solfile = get_new_filename(param.casename, ".jld2")
    #end

    res_io = nothing
    if log
        res_io = open(outfile*".txt", "w")
    end

    if increase_tau
        tau = hcat(tau, zeros(2*ngen+6*nline, iterlim))
    end

    if fts === nothing
        fts = Vector{Future}(undef, nprocs())
    end

    start_time = time()
    it = 0
    while it < iterlim
        it += 1
        info.it += 1

        param.u_prev .= param.u_curr
        param.v_prev .= param.v_curr
        param.l_prev .= param.l_curr

        parallel_solve_generators(fts, param, info)
        parallel_solve_branches(fts, param, info)
        parallel_solve_buses(fts, param, info)

        param.l_curr .+= param.rho .* (param.u_curr .- param.v_curr)
        param.rd .= -param.rho .* (param.v_curr .- param.v_prev)
        param.rp .= param.u_curr .- param.v_curr
        param.rp_old .= param.u_prev .- param.v_prev

        if ((info.it+param.Kf_mean-1) % param.Kf) == 0
            param.rp_k0 .= param.rp
        end

        info.primres = norm(param.rp)
        info.dualres = norm(param.rd)
        info.obj = get_objval(param, param.u_curr)
        info.gmax = get_gmax(param, param.u_curr, param.v_curr)

        info.rho_pq_max = rho_pq_max[it] = maximum(param.rho[1:2*ngen+4*nline])
        info.rho_pq_min = rho_pq_min[it] = minimum(param.rho[1:2*ngen+4*nline])
        info.rho_va_max = rho_va_max[it] = maximum(param.rho[2*ngen+4*nline+1:end])
        info.rho_va_min = rho_va_min[it] = minimum(param.rho[2*ngen+4*nline+1:end])

        eps_pri[it] = sqrt(length(param.l_curr))*ABSTOL + RELTOL*max(norm(param.u_curr), norm(-param.v_curr))
        eps_dual[it] = sqrt(length(param.u_curr))*ABSTOL + RELTOL*norm(param.l_curr)
        info.eps_pri = eps_pri[it]
        info.eps_dual = eps_dual[it]
        info.elapsed = time() - start_time
        if log
            print_iteration(info; res_io=res_io)
        end

        if info.primres < info.eps_pri && info.dualres < info.eps_dual
            info.term = :LocalOptimal
            break;
        end

        if savesol && (info.it % 100) == 0
            save_solution(solfile*".jld2", param, info)
        end

        if update_rho && info.it > 1
            adjust_rho(param, info, tau)
        end
    end
    info.elapsed = time() - start_time

    if log && final_message
        print_iteration_end_message(info; io=res_io)
    end
    if log
        close(res_io)
    end
    return param, info, tau
end



function admm_rect(optimizer_constructor; casename="case9", iterlim=100, eps=1e-4, rho_pq=400.0,
                   rho_va=40000.0, Kf=100, eta=0.01, update_rho=true, loadsol=false, savesol=false, log=false,
                   use_auglag=false, use_whole=false, linelimit=true)
    baseMVA = data.baseMVA
    buses = data.buses
    gens = data.generators
    lines = data.lines
    BusIdx = data.BusIdx
    BusGens = data.BusGenerators
    FromLines = data.FromLines
    ToLines = data.ToLines

    nbus = length(data.buses)
    ngen = length(data.generators)
    nline = length(data.lines)
    param, info = init_param(data, rho_pq, rho_va)
    param.Kf = Kf
    param.Kf_mean = 10
    param.eta = eta
    param.use_auglag = use_auglag
    param.use_whole = use_whole
    param.linelimit = linelimit
    param.optimizer_constructor = optimizer_constructor
    param.casename = casename

    info.it = 0
    info.term = :IterLimit

    # Remote references.
    fts = Vector{Future}(undef, nprocs())
    print_iteration_title()
    if loadsol
        load_solution(casename*".jld2", param, info)
    end

    outfile = get_new_filename(casename*"_result_nworkers"*string(nworkers()), ".txt")
    solfile = get_new_filename(casename, ".jld2")
    tau = zeros(2*ngen+6*nline, iterlim)
    # tau = zeros(2*ngen+6*nline, 500)
    tau[:,1] .= param.rho

    for p in workers()
        fts[p] = remotecall(build_branch_models, p, param)
    end
    for p in workers()
        fetch(fts[p])
    end

    initialize_branch_models(param)
    admm_rect_restart(param, info, tau; iterlim=iterlim, increase_tau=true, log=log, update_rho=update_rho, savesol=savesol, fts=fts, outfile=outfile, solfile=solfile)

    return param, info, tau
end
