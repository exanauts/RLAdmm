# ExaComDecOPF.jl

![Run tests](https://github.com/exanauts/ExaComDecOPF/workflows/Run%20tests/badge.svg)

This package implements a component-based decomposition of ACOPFs using ADMM in Julia
following the algorithm described in the
[paper](https://www.sciencedirect.com/science/article/abs/pii/S2352467718300389).
Bascially, there are three types of components -- generators, branches, and buses.
Each component defines a subproblem, and subproblems can be solved in parallel.
For example, if we have 10,000 branches, then 10,000 subproblems are defined, and
they can be solved in parallel.
The same applies to generators and buses.
We note that each subproblem is an NLP of very small size -- less than
10 variables and 10 constraints -- and is independent of the network size,
which makes the algorithm highly scalable.

The current implementation does not solve each subproblem in parallel
since Ipopt does not support parallelism.
However, the time result in the log will give approximate time to solve each
component.
An approximate elapsed time to solve the entire problem will be then
(iteration count) x (max time of each component solve)
if we were to run it in parallel.

## Installation

```julia
]add https://github.com/exanauts/ExaComDecOPF.jl
```

## How to run

```julia
$ export JULIA_PROJECT=${PWD}
$ julia -p n

julia> @everywhere using ExaComDecOPF
julia> @everywhere using Ipopt
julia> @everywhere ExaComDecOPF.load_case("data/case9")
julia> ExaComDecOPF.run_decomp(Ipopt.Optimizer; iterlim=500, rho_pq=400.0, rho_va=40000.0)
```

where n is the number of processes to use for parallel execution,
iterlim is the iteration limit, rho_pq is the penalty value for
the real and reactive power variables, and rho_va is the penalty value
for the voltage magnitude and angle variables.
It is assumed that the current directory specified by ${PWD} is the top
directory where ExaComDecOPF.jl package is located.

The [TRON](https://github.com/exanauts/Tron.git) solver can be used when you want to solve using augmented Lagrangian formulation.
In this case, you may want to execute as follows:

```julia
$ julia -p n

julia> @everywhere using ExaComDecOPF
julia> @everywhere using ExaTronInterface
julia> @everywhere ExaComDecOPF.load_case("data/case9")
julia> ExaComDecOPF.run_decomp(ExaTronInterface.Optimizer; iterlim=500, rho_pq=400.0, rho_va=40000.0, use_auglag=true, linelimit=false)
```

Currently, ADMM terminates when the primal and dual residuals become
less than predefined values, which are displayed in the log.

The following table gives the parameter values to use for each case.

| case | iterlim | rho_pq | rho_va |
| ---: | ------: | -----: | -----: |
|    9 |   1,000 |    400 | 40,000 |
|   14 |   5,000 |    400 |    400 |
|   30 |   5,000 |    400 | 40,000 |
|  118 |   1,000 |    400 |  4,000 |
|  300 |  10,000 |    400 |  4,000 |
|   89pegase | 10,000 | 100 | 10,000 |
| 1354pegase |  5,000 | 10 | 1,000 |
| 2869pegase | 10,000 | 10 | 1,000 |
| 9241pegase | 10,000 | 50 | 5,000 |
|13659pegase | 10,000 | 50 | 5,000 |

## Restart

In cases where you want to restart ExaComDecOPF from the last iteration on,
you can restart in the following way.
```julia
julia> param, info, tau = ExaComDecOPF.run_decomp(Ipopt.Optimizer; iteralim=1, rho_pq=400.0, rho_va=40000.0, final_message=false)
julia> param, info, tau = ExaComDecOPF.restart_decomp(param, info, tau; iterlim=1, final_message=false)
```

The first command will run one iteration of ADMM and return its state in `param, info` and `tau`.
The second command will resume the ADMM iteration from the state you specified and run one iteration again.
Note that you can turn on/off the final log message summarizing the iterations by setting `final_message=true/false`.
If you want to turn off the whole log message, then set an option `log=false`.

## Misc.

### Fixing `rho` values

Parameter `update_rho` determines whether to update `rho` value throughout the iteration. If you want to fix the value to the initially given value, then set it to `false` as follows:
```julia
ExaComDecOPF.run_decomp(Ipopt.Optimizer; iterlim=500, rho_pq=400.0, rho_va=40000.0, update_rho=false)
```

### Investigating primal/dual residuals and rho values

Using the returned `param`, you can investigate the primal/dual residuals and rho values.
* `param.rp`: primal residual
* `param.rd`: dual residual
* `param.rho`: rho value

Note that `param.rp` and `param.rd` are available after the first iteration and on.

