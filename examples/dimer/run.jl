using LinearAlgebra
BLAS.set_num_threads(1)
module DimerEngine
include(joinpath(@__DIR__, "../../simulations/mps/dimer/run_dimer_sector_cli.jl"))
end
include(joinpath(@__DIR__, "../mps_trace.jl"))
if abspath(PROGRAM_FILE) == @__FILE__
    AKLTMPSExample.main(DimerEngine.runy, "dimer",
        joinpath(@__DIR__, "../../simulations/mps/dimer/run_dimer_sector_cli.jl"))
end
