mutable struct AdmmParam
    N::Int

    rho_pq::Float64
    rho_va::Float64

    vm::Vector{Float64}
    va::Vector{Float64}
    vm_i_ij::Vector{Float64}
    va_i_ij::Vector{Float64}
    vm_j_ji::Vector{Float64}
    va_j_ji::Vector{Float64}

    pg::Vector{Float64}
    qg::Vector{Float64}
    pg_i::Vector{Float64}
    qg_i::Vector{Float64}

    pij::Vector{Float64}
    qij::Vector{Float64}
    pji::Vector{Float64}
    qji::Vector{Float64}
    pij_i::Vector{Float64}
    qij_i::Vector{Float64}
    pji_i::Vector{Float64}
    qji_i::Vector{Float64}

    lam_pg::Vector{Float64}
    lam_qg::Vector{Float64}
    lam_pij::Vector{Float64}
    lam_qij::Vector{Float64}
    lam_pji::Vector{Float64}
    lam_qji::Vector{Float64}
    lam_vm_i_ij::Vector{Float64}
    lam_va_i_ij::Vector{Float64}
    lam_vm_j_ji::Vector{Float64}
    lam_va_j_ji::Vector{Float64}

    optimizer_constructor

    function AdmmParam(ngen, nline, nbus, rho_pq, rho_va)
        param = new()
        param.rho_pq = rho_pq
        param.rho_va = rho_va
        param.vm = zeros(nbus)
        param.va = zeros(nbus)
        param.vm_i_ij = zeros(nline)
        param.va_i_ij = zeros(nline)
        param.vm_j_ji = zeros(nline)
        param.va_j_ji = zeros(nline)
        param.pg = zeros(ngen)
        param.qg = zeros(ngen)
        param.pg_i = zeros(ngen)
        param.qg_i = zeros(ngen)
        param.pij = zeros(nline)
        param.qij = zeros(nline)
        param.pji = zeros(nline)
        param.qji = zeros(nline)
        param.pij_i = zeros(nline)
        param.qij_i = zeros(nline)
        param.pji_i = zeros(nline)
        param.qji_i = zeros(nline)
        param.lam_pg = zeros(ngen)
        param.lam_qg = zeros(ngen)
        param.lam_pij = zeros(nline)
        param.lam_qij = zeros(nline)
        param.lam_pji = zeros(nline)
        param.lam_qji = zeros(nline)
        param.lam_vm_i_ij = zeros(nline)
        param.lam_va_i_ij = zeros(nline)
        param.lam_vm_j_ji = zeros(nline)
        param.lam_va_j_ji = zeros(nline)
        param.optimizer_constructor = nothing
        return param
    end
end

mutable struct AdmmInfo
    it::Int
    term::Symbol
    obj::Float64
    gmax::Float64
    primres::Float64
    dualres::Float64
    avg_gen::Float64
    avg_br::Float64
    avg_bus::Float64
    cum_avg_gen::Float64
    cum_avg_br::Float64
    cum_avg_bus::Float64
    elapsed::Float64

    function AdmmInfo()
        info = new()
        info.it = 0;
        info.term = :IterLimit
        info.obj = 0;
        info.gmax = Inf
        info.primres = Inf
        info.dualres = Inf
        info.avg_gen = 0
        info.avg_br = 0
        info.avg_bus = 0
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

function solve_generators(param::AdmmParam, info::AdmmInfo)
    baseMVA = data.baseMVA
    gens = data.generators
    ngen = length(data.generators)

    p = myid()
    nele = floor(Int, ngen / nworkers())
    nrem = mod(ngen, nworkers())
    start = get_start(p, nele, nrem)
    last = start + get_inc(p, nele, nrem)

    t = 0.0
    for g=start:last
        # Closed-form solution exist.
        t += @elapsed begin
            param.pg[g] = max(gens[g].Pmin,
                              min(gens[g].Pmax,
                                  (-(gens[g].coeff[gens[g].n-1]*baseMVA + param.lam_pg[g] - param.rho_pq*param.pg_i[g])) / (2*gens[g].coeff[gens[g].n-2]*(baseMVA^2) + param.rho_pq)))
            param.qg[g] = max(gens[g].Qmin, min(gens[g].Qmax, (-(param.lam_qg[g] - param.rho_pq*param.qg_i[g])) / param.rho_pq))
        end
        #=
        m = Model(solver=IpoptSolver())

        @variable(m, gens[g].Pmin <= pg <= gens[g].Pmax, start=param.pg[g])
        @variable(m, gens[g].Qmin <= qg <= gens[g].Qmax, start=param.qg[g])
        @NLobjective(m, Min,
                        gens[g].coeff[gens[g].n-2]*(baseMVA*pg)^2 +
                        gens[g].coeff[gens[g].n-1]*(baseMVA*pg) +
                        gens[g].coeff[gens[g].n] +
                        param.lam_pg[g]*pg +
                        param.lam_qg[g]*qg +
                        0.5*param.rho_pq*(
                            (pg - param.pg_i[g])^2 +
                            (qg - param.qg_i[g])^2
                        )
        )

        t += @elapsed st = solve(m)
        if st != :Optimal
            println("Generator ", g, " does not solve to optimality: ", st)
            @assert(false)
        end

        param.pg[g] = getvalue(m[:pg])
        param.qg[g] = getvalue(m[:qg])
        =#
    end

    if start <= last
        info.avg_gen = t / (last-start+1)
        info.cum_avg_gen += info.avg_gen
    end

    return param, info
end

function solve_branches(param::AdmmParam, info::AdmmInfo)
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

    t = 0.0
    for l=start:last
        fr = BusIdx[lines[l].from]
        to = BusIdx[lines[l].to]

        m = Model(param.optimizer_constructor)

        @variable(m, pij, start=param.pij[l])
        @variable(m, qij, start=param.qij[l])
        @variable(m, buses[fr].Vmin <= vm_i_ij <= buses[fr].Vmax, start=param.vm_i_ij[l])
        @variable(m, va_i_ij, start=param.va_i_ij[l])
        @variable(m, pji, start=param.pji[l])
        @variable(m, qji, start=param.qji[l])
        @variable(m, buses[to].Vmin <= vm_j_ji <= buses[to].Vmax, start=param.vm_j_ji[l])
        @variable(m, va_j_ji, start=param.va_j_ji[l])

        if fr == busref
            setlowerbound(va_i_ij, buses[busref].Va)
            setupperbound(va_i_ij, buses[busref].Va)
        end

        if to == busref
            setlowerbound(va_j_ji, buses[busref].Va)
            setupperbound(va_j_ji, buses[busref].Va)
        end

        @NLobjective(m, Min,
                        param.lam_pij[l]*pij +
                        param.lam_qij[l]*qij +
                        param.lam_vm_i_ij[l]*vm_i_ij +
                        param.lam_va_i_ij[l]*va_i_ij +
                        param.lam_pji[l]*pji +
                        param.lam_qji[l]*qji +
                        param.lam_vm_j_ji[l]*vm_j_ji +
                        param.lam_va_j_ji[l]*va_j_ji +
                        0.5*param.rho_va*(
                            (vm_i_ij - param.vm[fr])^2 +
                            (va_i_ij - param.va[fr])^2 +
                            (vm_j_ji - param.vm[to])^2 +
                            (va_j_ji - param.va[to])^2
                        ) +
                        0.5*param.rho_pq*(
                            (pij - param.pij_i[l])^2 +
                            (qij - param.qij_i[l])^2 +
                            (pji - param.pji_i[l])^2 +
                            (qji - param.qji_i[l])^2
                        )
        )

        @NLconstraint(m, pij == YffR[l]*(vm_i_ij)^2 +
                                vm_i_ij*vm_j_ji*(
                                    YftR[l]*cos(va_i_ij - va_j_ji) +
                                    YftI[l]*sin(va_i_ij - va_j_ji)
                                )
        )

        @NLconstraint(m, qij == -YffI[l]*(vm_i_ij)^2 +
                                vm_i_ij*vm_j_ji*(
                                    -YftI[l]*cos(va_i_ij - va_j_ji) +
                                    YftR[l]*sin(va_i_ij - va_j_ji)
                                )
        )

        @NLconstraint(m, pji == YttR[l]*(vm_j_ji)^2 +
                                vm_i_ij*vm_j_ji*(
                                    YtfR[l]*cos(va_j_ji - va_i_ij) +
                                    YtfI[l]*sin(va_j_ji - va_i_ij)
                                )
        )

        @NLconstraint(m, qji == -YttI[l]*(vm_j_ji)^2 +
                                vm_i_ij*vm_j_ji*(
                                    -YtfI[l]*cos(va_j_ji - va_i_ij) +
                                    YtfR[l]*sin(va_j_ji - va_i_ij)
                                )
        )

        if lines[l].rateA > 0 && lines[l].rateA < 1e10
            @NLconstraint(m, pij^2 + qij^2 <= (lines[l].rateA / baseMVA)^2)
            @NLconstraint(m, pji^2 + qji^2 <= (lines[l].rateA / baseMVA)^2)
        end

        t += @elapsed st = solve(m)
        if st != :Optimal
            println("Branch ", l, " does not solve to optimality: ", st)
            @assert(false)
        end

        param.pij[l] = getvalue(m[:pij])
        param.qij[l] = getvalue(m[:qij])
        param.pji[l] = getvalue(m[:pji])
        param.qji[l] = getvalue(m[:qji])
        param.vm_i_ij[l] = getvalue(m[:vm_i_ij])
        param.vm_j_ji[l] = getvalue(m[:vm_j_ji])
        param.va_i_ij[l] = getvalue(m[:va_i_ij])
        param.va_j_ji[l] = getvalue(m[:va_j_ji])
    end

    if start <= last
        info.avg_br = t / (last-start+1)
        info.cum_avg_br += info.avg_br
    end

    return param, info
end

function solve_buses(param::AdmmParam, info::AdmmInfo)
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

    t = 0
    for b=start:last
        m = Model(param.optimizer_constructor)

        @variable(m, buses[b].Vmin <= vm <= buses[b].Vmax, start=param.vm[b])
        @variable(m, va, start=param.va[b])
        @variable(m, gens[g].Pmin <= pg_i[g in BusGens[b]] <= gens[g].Pmax, start=param.pg_i[g])
        @variable(m, gens[g].Qmin <= qg_i[g in BusGens[b]] <= gens[g].Qmax, start=param.qg_i[g])
        @variable(m, pij_i[l in FromLines[b]], start=param.pij_i[l])
        @variable(m, qij_i[l in FromLines[b]], start=param.qij_i[l])
        @variable(m, pji_i[l in ToLines[b]], start=param.pji_i[l])
        @variable(m, qji_i[l in ToLines[b]], start=param.qji_i[l])

        if b == busref
            setlowerbound(va, buses[busref].Va)
            setlowerbound(va, buses[busref].Va)
        end

        @NLobjective(m, Min,
                        sum(-param.lam_pg[g]*pg_i[g] for g in BusGens[b]) +
                        sum(-param.lam_qg[g]*qg_i[g] for g in BusGens[b]) +
                        sum(-param.lam_pij[l]*pij_i[l] for l in FromLines[b]) +
                        sum(-param.lam_qij[l]*qij_i[l] for l in FromLines[b]) +
                        sum(-param.lam_pji[l]*pji_i[l] for l in ToLines[b]) +
                        sum(-param.lam_qji[l]*qji_i[l] for l in ToLines[b]) +
                        sum(-param.lam_vm_i_ij[l]*vm for l in FromLines[b]) +
                        sum(-param.lam_va_i_ij[l]*va for l in FromLines[b]) +
                        sum(-param.lam_vm_j_ji[l]*vm for l in ToLines[b]) +
                        sum(-param.lam_va_j_ji[l]*va for l in ToLines[b]) +
                        0.5*param.rho_pq*(
                            sum((param.pg[g] - pg_i[g])^2 +
                                (param.qg[g] - qg_i[g])^2 for g in BusGens[b])
                        ) +
                        0.5*param.rho_pq*(
                            sum((param.pij[l] - pij_i[l])^2 +
                                (param.qij[l] - qij_i[l])^2
                                for l in FromLines[b]) +
                            sum((param.pji[l] - pji_i[l])^2 +
                                (param.qji[l] - qji_i[l])^2
                                for l in ToLines[b])
                        ) +
                        0.5*param.rho_va*(
                            sum((param.vm_i_ij[l] - vm)^2 +
                                (param.va_i_ij[l] - va)^2
                                for l in FromLines[b]) +
                            sum((param.vm_j_ji[l] - vm)^2 +
                                (param.va_j_ji[l] - va)^2
                                for l in ToLines[b])
                        )
        )

        @constraint(m,
                   (sum(baseMVA*pg_i[g] for g in BusGens[b]) - buses[b].Pd) / baseMVA
                   == sum(pij_i[l] for l in FromLines[b]) +
                      sum(pji_i[l] for l in ToLines[b]) +
                      YshR[b]*(vm)^2
        )

        @constraint(m,
                   (sum(baseMVA*qg_i[g] for g in BusGens[b]) - buses[b].Qd) / baseMVA
                   == sum(qij_i[l] for l in FromLines[b]) +
                      sum(qji_i[l] for l in ToLines[b]) -
                      YshI[b]*(vm)^2
        )

        t += @elapsed st = solve(m)
        if st != :Optimal
            println("Bus ", b, " does not solve to optimality: ", st)
            @assert(false)
        end

        info.dualres = max(info.dualres,
                           param.rho_va*abs(param.vm[b] - getvalue(m[:vm])),
                           param.rho_va*abs(param.va[b] - getvalue(m[:va]))
        )
        param.vm[b] = getvalue(m[:vm])
        param.va[b] = getvalue(m[:va])
        for g in BusGens[b]
            info.dualres = max(info.dualres,
                               param.rho_pq*abs(param.pg_i[g] - getvalue(m[:pg_i][g])),
                               param.rho_pq*abs(param.qg_i[g] - getvalue(m[:qg_i][g]))
            )
            param.pg_i[g] = getvalue(m[:pg_i][g])
            param.qg_i[g] = getvalue(m[:qg_i][g])
        end
        for l in FromLines[b]
            info.dualres = max(info.dualres,
                               param.rho_pq*abs(param.pij_i[l] - getvalue(m[:pij_i][l])),
                               param.rho_pq*abs(param.qij_i[l] - getvalue(m[:qij_i][l])),
            )
            param.pij_i[l] = getvalue(m[:pij_i][l])
            param.qij_i[l] = getvalue(m[:qij_i][l])
        end
        for l in ToLines[b]
            info.dualres = max(info.dualres,
                               param.rho_pq*abs(param.pji_i[l] - getvalue(m[:pji_i][l])),
                               param.rho_pq*abs(param.qji_i[l] - getvalue(m[:qji_i][l]))
            )
            param.pji_i[l] = getvalue(m[:pji_i][l])
            param.qji_i[l] = getvalue(m[:qji_i][l])
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

    param = AdmmParam(ngen, nline, nbus, rho_pq, rho_va)
    info = AdmmInfo()

    for b=1:nbus
        param.vm[b] = 1.0
    end

    for g=1:ngen
        param.pg[g] = (gens[g].Pmin + gens[g].Pmax) / 2
        param.qg[g] = (gens[g].Qmin + gens[g].Qmax) / 2
    end

    return param, info
end

function load_param(data::OPFData, param::AdmmParam, sol::Dict{String,Any})
    nbus = length(data.buses)
    ngen = length(data.generators)
    nline = length(data.lines)

    param.rho_pq = sol["rho_pq"]
    param.rho_va = sol["rho_va"]
    copyto!(param.vm, 1, sol["vm"], 1, nbus)
    copyto!(param.va, 1, sol["va"], 1, nbus)
    copyto!(param.vm_i_ij, 1, sol["vm_i_ij"], 1, nline)
    copyto!(param.va_i_ij, 1, sol["va_i_ij"], 1, nline)
    copyto!(param.vm_j_ji, 1, sol["vm_j_ji"], 1, nline)
    copyto!(param.va_j_ji, 1, sol["va_j_ji"], 1, nline)
    copyto!(param.pg, 1, sol["pg"], 1, ngen)
    copyto!(param.qg, 1, sol["qg"], 1, ngen)
    copyto!(param.pg_i, 1, sol["pg_i"], 1, ngen)
    copyto!(param.qg_i, 1, sol["qg_i"], 1, ngen)
    copyto!(param.pij, 1, sol["pij"], 1, nline)
    copyto!(param.qij, 1, sol["qij"], 1, nline)
    copyto!(param.pji, 1, sol["pji"], 1, nline)
    copyto!(param.qji, 1, sol["qji"], 1, nline)
    copyto!(param.pij_i, 1, sol["pij_i"], 1, nline)
    copyto!(param.qij_i, 1, sol["qij_i"], 1, nline)
    copyto!(param.pji_i, 1, sol["pji_i"], 1, nline)
    copyto!(param.qji_i, 1, sol["qji_i"], 1, nline)
    copyto!(param.lam_pg, 1, sol["lam_pg"], 1, ngen)
    copyto!(param.lam_qg, 1, sol["lam_qg"], 1, ngen)
    copyto!(param.lam_pij, 1, sol["lam_pij"], 1, nline)
    copyto!(param.lam_qij, 1, sol["lam_qij"], 1, nline)
    copyto!(param.lam_pji, 1, sol["lam_pji"], 1, nline)
    copyto!(param.lam_qji, 1, sol["lam_qji"], 1, nline)
    copyto!(param.lam_vm_i_ij, 1, sol["lam_vm_i_ij"], 1, nline)
    copyto!(param.lam_va_i_ij, 1, sol["lam_va_i_ij"], 1, nline)
    copyto!(param.lam_vm_j_ji, 1, sol["lam_vm_j_ji"], 1, nline)
    copyto!(param.lam_va_j_ji, 1, sol["lam_va_j_ji"], 1, nline)
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

function parallel_solve_generators(fts::Vector{Future}, param::AdmmParam, info::AdmmInfo)
    # Solve generators in parallel in a distributed fashion.
    for p in workers()
        fts[p] = remotecall(solve_generators, p, param, info)
    end

    # Collect the results.
    ngen = length(data.generators)
    nele = floor(Int, ngen / nworkers())
    nrem = mod(ngen, nworkers())
    avg_gen = 0

    for p in workers()
        w_param, w_info = fetch(fts[p])
        start = get_start(p, nele, nrem)
        last = start + get_inc(p, nele, nrem)
        if start <= last
            for g=start:last
                param.pg[g] = w_param.pg[g]
                param.qg[g] = w_param.qg[g]
            end
            avg_gen += w_info.avg_gen * (last-start+1)
        end
    end
    info.avg_gen = avg_gen / ngen
    info.cum_avg_gen += info.avg_gen
end

function parallel_solve_branches(fts::Vector{Future}, param::AdmmParam, info::AdmmInfo)
    # Solve branches in parallel in a distributed fashion.
    for p in workers()
        fts[p] = remotecall(solve_branches, p, param, info)
    end

    # Collect the results.
    nline = length(data.lines)
    nele = floor(Int, nline / nworkers())
    nrem = mod(nline, nworkers())
    avg_br = 0

    for p in workers()
        w_param, w_info = fetch(fts[p])
        start = get_start(p, nele, nrem)
        last = start + get_inc(p, nele, nrem)
        if start <= last
            for l=start:last
                param.pij[l] = w_param.pij[l]
                param.qij[l] = w_param.qij[l]
                param.pji[l] = w_param.pji[l]
                param.qji[l] = w_param.qji[l]
                param.vm_i_ij[l] = w_param.vm_i_ij[l]
                param.vm_j_ji[l] = w_param.vm_j_ji[l]
                param.va_i_ij[l] = w_param.va_i_ij[l]
                param.va_j_ji[l] = w_param.va_j_ji[l]
            end
            avg_br += w_info.avg_br * (last-start+1)
        end
    end
    info.avg_br = avg_br / nline
    info.cum_avg_br += info.avg_br
end

function parallel_solve_buses(fts::Vector{Future}, param::AdmmParam, info::AdmmInfo)
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

    for p in workers()
        w_param, w_info = fetch(fts[p])
        start = get_start(p, nele, nrem)
        last = start + get_inc(p, nele, nrem)
        if start <= last
            for b=start:last
                param.vm[b] = w_param.vm[b]
                param.va[b] = w_param.va[b]
                for g in BusGens[b]
                    param.pg_i[g] = w_param.pg_i[g]
                    param.qg_i[g] = w_param.qg_i[g]
                end
                for l in FromLines[b]
                    param.pij_i[l] = w_param.pij_i[l]
                    param.qij_i[l] = w_param.qij_i[l]
                end
                for l in ToLines[b]
                    param.pji_i[l] = w_param.pji_i[l]
                    param.qji_i[l] = w_param.qji_i[l]
                end
            end
            avg_bus += w_info.avg_bus * (last-start+1)
            info.dualres = max(info.dualres, w_info.dualres)
        end
    end
    info.avg_bus = avg_bus / nbus
    info.cum_avg_bus += info.avg_bus
end

function admm_polar(optimizer_constructor; casename="case9", iterlim=100, eps=1e-4, rho_pq=400.0,
                      rho_va=40000.0, loadsol=false)
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
    param.optimizer_constructor = optimizer_constructor
    info.it = 0
    info.term = :IterLimit

    # Remote references.
    fts = Vector{Future}(undef, nprocs())

    @printf("%10s  %12s  %12s  %12s  %12s  %8s  %8s  %8s  %12s\n",
            "Iterations", "PrimRes", "DualRes", "Objective", "||g||",
            "GenTime", "BrTime", "BusTime", "Elapsed")
    if loadsol
        sol = load(casename*".jld2")
        info.it = sol["it"]
        info.gmax = sol["gmax"]
        info.obj = sol["obj"]
        info.primres = sol["primres"]
        info.dualres = sol["dualres"]
        load_param(data, param, sol)
    end

    outfile = get_new_filename(casename*"_result_nworkers"*string(nworkers()), ".txt")
    solfile = get_new_filename(casename, ".jld2")
    res_io = open(outfile*".txt", "w")

    start_time = time()
    while info.it <= iterlim && info.gmax > eps
        info.it += 1

        # Dual residual is computed in solve_buses().
        info.primres = 0
        info.dualres = 0

        parallel_solve_generators(fts, param, info)
        parallel_solve_branches(fts, param, info)
        parallel_solve_buses(fts, param, info)

        for g=1:ngen
            param.lam_pg[g] += param.rho_pq*(param.pg[g] - param.pg_i[g])
            param.lam_qg[g] += param.rho_pq*(param.qg[g] - param.qg_i[g])
            info.primres = max(info.primres,
                               abs(param.pg[g] - param.pg_i[g]),
                               abs(param.qg[g] - param.qg_i[g])
            )
        end

        for l=1:nline
            param.lam_pij[l] += param.rho_pq*(param.pij[l] - param.pij_i[l])
            param.lam_qij[l] += param.rho_pq*(param.qij[l] - param.qij_i[l])
            param.lam_pji[l] += param.rho_pq*(param.pji[l] - param.pji_i[l])
            param.lam_qji[l] += param.rho_pq*(param.qji[l] - param.qji_i[l])
            param.lam_vm_i_ij[l] += param.rho_va*(param.vm_i_ij[l] - param.vm[BusIdx[lines[l].from]])
            param.lam_va_i_ij[l] += param.rho_va*(param.va_i_ij[l] - param.va[BusIdx[lines[l].from]])
            param.lam_vm_j_ji[l] += param.rho_va*(param.vm_j_ji[l] - param.vm[BusIdx[lines[l].to]])
            param.lam_va_j_ji[l] += param.rho_va*(param.va_j_ji[l] - param.va[BusIdx[lines[l].to]])
            info.primres = max(info.primres,
                               abs(param.pij[l] - param.pij_i[l]),
                               abs(param.qij[l] - param.qij_i[l]),
                               abs(param.pji[l] - param.pji_i[l]),
                               abs(param.qji[l] - param.qji_i[l]),
                               abs(param.vm_i_ij[l] - param.vm[BusIdx[lines[l].from]]),
                               abs(param.va_i_ij[l] - param.va[BusIdx[lines[l].from]]),
                               abs(param.vm_j_ji[l] - param.vm[BusIdx[lines[l].to]]),
                               abs(param.va_j_ji[l] - param.va[BusIdx[lines[l].to]]),
            )
        end

        info.obj = sum(gens[g].coeff[gens[g].n-2]*(baseMVA*param.pg[g])^2 +
                       gens[g].coeff[gens[g].n-1]*(baseMVA*param.pg[g]) +
                       gens[g].coeff[gens[g].n]
                       for g in 1:ngen)

        info.gmax = 0
        for b=1:nbus
            pbal = 0
            qbal = 0

            for g in BusGens[b]
                pbal += baseMVA*param.pg[g]
                qbal += baseMVA*param.qg[g]
            end

            pbal -= buses[b].Pd
            qbal -= buses[b].Qd
            pbal /= baseMVA
            qbal /= baseMVA

            for l in FromLines[b]
                pbal -= param.pij[l]
                qbal -= param.qij[l]
            end

            for l in ToLines[b]
                pbal -= param.pji[l]
                qbal -= param.qji[l]
            end

            pbal -= ybus.YshR[b]*(param.vm[b])^2
            qbal += ybus.YshI[b]*(param.vm[b])^2
            info.gmax = max(info.gmax, abs(pbal), abs(qbal))
        end

        for l=1:nline
            fr = BusIdx[lines[l].from]
            to = BusIdx[lines[l].to]
            info.gmax = max(info.gmax,
                            abs(param.vm[fr] - param.vm_i_ij[l]),
                            abs(param.va[fr] - param.va_i_ij[l]),
                            abs(param.vm[to] - param.vm_j_ji[l]),
                            abs(param.va[to] - param.va_j_ji[l])
            )
        end

        if (info.it % 50) == 0
            @printf("%10s  %12s  %12s  %12s  %12s  %8s  %8s  %8s  %12s\n",
                    "Iterations", "PrimRes", "DualRes", "Objective", "||g||",
                    "GenTime", "BrTime", "BusTime", "Elapsed")
        end
        #=
        if (info.it % 5) == 0
            save(solfile*".jld2",
                "it", info.it, "gmax", info.gmax, "obj", info.obj,
                "primres", info.primres, "dualres", info.dualres,
                "rho_pq", param.rho_pq, "rho_va", param.rho_va,
                "vm", param.vm, "va", param.va,
                "vm_i_ij", param.vm_i_ij, "va_i_ij", param.va_i_ij,
                "vm_j_ji", param.vm_j_ji, "va_j_ji", param.va_j_ji,
                "pg", param.pg, "qg", param.qg,
                "pg_i", param.pg_i, "qg_i", param.qg_i,
                "pij", param.pij, "qij", param.qij,
                "pji", param.pji, "qji", param.qji,
                "pij_i", param.pij_i, "qij_i", param.qij_i,
                "pji_i", param.pji_i, "qji_i", param.qji_i,
                "lam_pg", param.lam_pg, "lam_qg", param.lam_qg,
                "lam_pij", param.lam_pij, "lam_qij", param.lam_qij,
                "lam_pji", param.lam_pji, "lam_qji", param.lam_qji,
                "lam_vm_i_ij", param.lam_vm_i_ij, "lam_va_i_ij", param.lam_va_i_ij,
                "lam_vm_j_ji", param.lam_vm_j_ji, "lam_va_j_ji", param.lam_va_j_ji)
        end
        =#
        info.elapsed = time() - start_time
        @printf("%10d  %.6e  %.6e  %.6e  %.6e  %8.4f  %8.4f  %8.4f  %12.4f\n",
                info.it, info.primres, info.dualres, info.obj, info.gmax,
                info.avg_gen, info.avg_br, info.avg_bus, info.elapsed)
        @printf(res_io, "%10d  %.6e  %.6e  %.6e  %.6e  %12.4f\n", info.it, info.primres,
                info.dualres, info.obj, info.gmax, info.elapsed)
        flush(res_io)
    end
    info.elapsed = time() - start_time
    close(res_io)

    if info.gmax <= eps
        info.term = :LocalOptimal
    end

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
    @printf("Elapsed time (secs). . . . . . . . . % 12.3f\n", info.elapsed)
    @printf("Average parallel generators (secs) . % 12.3f\n", info.cum_avg_gen / info.it)
    @printf("Average parallel branches (secs) . . % 12.3f\n", info.cum_avg_br / info.it)
    @printf("Average parallel buses (secs)  . . . % 12.3f\n", info.cum_avg_bus / info.it)
end
