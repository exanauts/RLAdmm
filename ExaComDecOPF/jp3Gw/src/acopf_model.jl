# Define a monolithic ACOPF model.
function acopf_model(data::OPFData, ybus::Ybus)
    m = Model()

    nbus = length(data.buses)
    ngen = length(data.generators)
    nline = length(data.lines)

    baseMVA = data.baseMVA
    buses = data.buses
    busref = data.bus_ref
    gens = data.generators
    lines = data.lines
    BusIdx = data.BusIdx
    BusGens = data.BusGenerators
    FromLines = data.FromLines
    ToLines = data.ToLines

    linelim_idx = findall(x -> x.rateA > 0 && x.rateA < 1e10, lines)
    LimIdx = zeros(Int, length(linelim_idx))
    for (k,v) in enumerate(linelim_idx)
        LimIdx[k] = v
    end
    nlimit = length(LimIdx)

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    @variable(m, buses[b].Vmin <= vm[b=1:nbus] <= buses[b].Vmax)
    @variable(m, va[b=1:nbus])
    @variable(m, gens[g].Pmin <= pg[g=1:ngen] <= gens[g].Pmax)
    @variable(m, gens[g].Qmin <= qg[g=1:ngen] <= gens[g].Qmax)
    @variable(m, pij[l=1:nline])
    @variable(m, qij[l=1:nline])
    @variable(m, pji[l=1:nline])
    @variable(m, qji[l=1:nline])

    #=
    This makes objective value larger.
    setlowerbound(vm[busref], buses[busref].Vm)
    setupperbound(vm[busref], buses[busref].Vm)
    =#
    setlowerbound(va[busref], buses[busref].Va)
    setupperbound(va[busref], buses[busref].Va)

    @NLobjective(m, Min,
                    sum(gens[g].coeff[gens[g].n-2]*(baseMVA*pg[g])^2 +
                        gens[g].coeff[gens[g].n-1]*(baseMVA*pg[g]) +
                        gens[g].coeff[gens[g].n] for g in 1:ngen)
    )

    @constraint(m, real_balance[b=1:nbus],
                (sum(baseMVA*pg[g] for g in BusGens[b]) - buses[b].Pd) / baseMVA
                == sum(pij[l] for l in FromLines[b]) +
                   sum(pji[l] for l in ToLines[b]) +
                   YshR[b]*(vm[b])^2
    )

    @constraint(m, reactive_balance[b=1:nbus],
                (sum(baseMVA*qg[g] for g in BusGens[b]) - buses[b].Qd) / baseMVA
                == sum(qij[l] for l in FromLines[b]) +
                   sum(qji[l] for l in ToLines[b]) -
                   YshI[b]*(vm[b])^2
    )

    @NLconstraint(m, real_fromto[l=1:nline],
                     pij[l] == YffR[l]*(vm[BusIdx[lines[l].from]])^2 +
                               vm[BusIdx[lines[l].from]]*vm[BusIdx[lines[l].to]]*(
                                   YftR[l]*cos(va[BusIdx[lines[l].from]] - va[BusIdx[lines[l].to]]) +
                                   YftI[l]*sin(va[BusIdx[lines[l].from]] - va[BusIdx[lines[l].to]])
                               )
    )

    @NLconstraint(m, reactive_fromto[l=1:nline],
                     qij[l] == -YffI[l]*(vm[BusIdx[lines[l].from]])^2 +
                               vm[BusIdx[lines[l].from]]*vm[BusIdx[lines[l].to]]*(
                                   -YftI[l]*cos(va[BusIdx[lines[l].from]] - va[BusIdx[lines[l].to]]) +
                                   YftR[l]*sin(va[BusIdx[lines[l].from]] - va[BusIdx[lines[l].to]])
                               )
    )

    @NLconstraint(m, real_tofrom[l=1:nline],
                     pji[l] == YttR[l]*(vm[BusIdx[lines[l].to]])^2 +
                               vm[BusIdx[lines[l].to]]*vm[BusIdx[lines[l].from]]*(
                                   YtfR[l]*cos(va[BusIdx[lines[l].to]] - va[BusIdx[lines[l].from]]) +
                                   YtfI[l]*sin(va[BusIdx[lines[l].to]] - va[BusIdx[lines[l].from]])
                               )
    )

    @NLconstraint(m, reactive_tofrom[l=1:nline],
                     qji[l] == -YttI[l]*(vm[BusIdx[lines[l].to]])^2 +
                               vm[BusIdx[lines[l].to]]*vm[BusIdx[lines[l].from]]*(
                                   -YtfI[l]*cos(va[BusIdx[lines[l].to]] - va[BusIdx[lines[l].from]]) +
                                   YtfR[l]*sin(va[BusIdx[lines[l].to]] - va[BusIdx[lines[l].from]])
                               )
    )

    @NLconstraint(m, from_limit[l=1:nlimit],
                     pij[LimIdx[l]]^2 + qij[LimIdx[l]]^2 <= (lines[LimIdx[l]].rateA / baseMVA)^2
    )

    @NLconstraint(m, to_limit[l=1:nlimit],
                     pji[LimIdx[l]]^2 + qji[LimIdx[l]]^2 <= (lines[LimIdx[l]].rateA / baseMVA)^2
    )

    return m
end
