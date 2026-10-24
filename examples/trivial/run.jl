using LinearAlgebra
BLAS.set_num_threads(1)
module TrivialEngine
include(joinpath(@__DIR__, "../../simulations/mps/trivial/run_zero_static.jl"))
end
include(joinpath(@__DIR__, "../mps_trace.jl"))
if abspath(PROGRAM_FILE) == @__FILE__
    AKLTMPSExample.main(TrivialEngine.runy, "trivial",
        joinpath(@__DIR__, "../../simulations/mps/trivial/run_zero_static.jl"))
end
