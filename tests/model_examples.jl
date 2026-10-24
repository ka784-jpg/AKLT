using Test, DelimitedFiles, TOML
include(joinpath(@__DIR__, "../examples/mps_trace.jl"))
using .AKLTMPSExample

# Synthetic outputs test wrapper contracts; the CI MPS runs test the real engines.
function fixture_runner(L, p, cutoff, sweeps, iterations, periodic, origins, ti, tf, interval, seed, prefix, extra...)
    suffix = isempty(extra) ? "zero" : "dimer"
    function emit(name, columns, value)
        values = fill(NaN, sweeps+1, columns+1)
        values[1, 2:end] .= 42
        values[2:end, 1] .= 1:sweeps
        values[2:end, 2:end] .= value
        writedlm(prefix * name * "_QCMPS_" * suffix, values, ',')
    end
    emit("_SO_NN", origins, -0.2)
    emit("_S", 1, 1.0)
    suffix == "dimer" && emit("_Dimer", 1, -0.3)
end

@testset "MPS example interface contracts (synthetic outputs)" begin
    mktempdir() do tmp
        for model in ("trivial", "dimer")
            config = joinpath(@__DIR__, "../examples", model, "config.toml")
            output = joinpath(tmp, model)
            AKLTMPSExample.run_demo(fixture_runner, model, config, output, @__FILE__)
            values, header = readdlm(joinpath(output, "summary.csv"), ',', Float64; header=true)
            @test size(values, 1) == 16
            @test all(values[:,2] .== -0.2)
            @test values[:,3] == abs.(values[:,2])
            @test all(values[:,4] .== 1)
            @test TOML.parsefile(joinpath(output,"metadata.toml"))["trajectory_seed"] == 42
            if model == "dimer"
                @test values[:,6] == abs.(values[:,5])
            end
            old = read(joinpath(output, "summary.csv"))
            @test_throws ArgumentError AKLTMPSExample.run_demo(fixture_runner, model, config, output, @__FILE__)
            @test old == read(joinpath(output, "summary.csv"))
            @test_throws ErrorException AKLTMPSExample.run_demo((args...)->nothing, model, config, joinpath(tmp,"missing_$model"), @__FILE__)
            @test !ispath(joinpath(tmp,"missing_$model"))
        end
        @test isempty(filter(n -> startswith(n,".aklt-mps-example-"), readdir(tmp)))
        base = TOML.parsefile(joinpath(@__DIR__, "../examples/trivial/config.toml"))
        for (key, value) in (("L", 8), ("L", true), ("sweeps", 41), ("seed", -1),
            ("p_competing", NaN), ("p_competing", 1.1), ("cutoff", 0), ("unknown", 1))
            path = joinpath(tmp, "bad.toml")
            open(io -> TOML.print(io, merge(base, Dict(key=>value))), path, "w")
            @test_throws ArgumentError AKLTMPSExample.read_config(path)
        end
    end
end
