using Test, LinearAlgebra, DelimitedFiles
BLAS.set_num_threads(1)

module TrivialRun
include(joinpath(@__DIR__, "../simulations/mps/trivial/run_zero_static.jl"))
end
module DimerRun
include(joinpath(@__DIR__, "../simulations/mps/dimer/run_dimer_sector_cli.jl"))
end

@testset "Small MPS trajectories" begin
    mktempdir() do tmp
        TrivialRun.runy(4, 0.5, 1e-10, 4, 1, true, 2, 2, 4, 1, 1234, joinpath(tmp,"tt"))
        DimerRun.runy(4, 0.66, 1e-10, 4, 1, true, 2, 2, 4, 1, 1234, joinpath(tmp,"dimer"), "odd", "all", "half", 0)
        files = filter(p -> filesize(p) > 0, readdir(tmp; join=true))
        @test any(p -> startswith(basename(p), "tt"), files)
        @test any(p -> startswith(basename(p), "dimer"), files)
        for file in files
            values = readdlm(file, ',', Float64)
            @test all(isfinite, values[2:end,2:end])
        end
    end
end
