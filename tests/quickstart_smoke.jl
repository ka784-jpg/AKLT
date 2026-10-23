using Test, DelimitedFiles, TOML
include(joinpath(@__DIR__, "../examples/quickstart.jl"))
using .AKLTQuickstart

const CONFIG = joinpath(@__DIR__, "../examples/quickstart.toml")

@testset "Quickstart interface and reproducible output" begin
    mktempdir() do tmp
        first = AKLTQuickstart.run_demo(CONFIG, joinpath(tmp, "first"))
        second = AKLTQuickstart.run_demo(CONFIG, joinpath(tmp, "second"))
        @test isdir(joinpath(first, "raw"))
        for name in ("samples.csv", "summary.csv", "metadata.toml")
            @test read(joinpath(first, name)) == read(joinpath(second, name))
        end
        summary, header = readdlm(joinpath(first, "summary.csv"), ',', Float64; header=true)
        expected, expected_header = readdlm(joinpath(@__DIR__, "../examples/expected/summary.csv"), ',', Float64; header=true)
        @test header == expected_header
        @test summary ≈ expected rtol=1e-9 atol=1e-11
        @test size(summary) == (5, 5)
        @test summary[:,1] == 20:24
        @test summary[:,4:5] == abs.(summary[:,2:3])
        samples = readdlm(joinpath(first, "samples.csv"), ',', Float64; skipstart=1)
        @test size(samples) == (25, 7)
        @test all(isfinite, samples)
        @test all(==(1234), samples[:,3])
        @test TOML.parsefile(joinpath(first, "metadata.toml"))["parameters"] == TOML.parsefile(CONFIG)
        @test_throws ArgumentError AKLTQuickstart.run_demo(CONFIG, first)
        @test read(joinpath(first, "summary.csv")) == read(joinpath(second, "summary.csv"))
        existing_file = joinpath(tmp, "file")
        write(existing_file, "preserve me")
        @test_throws ArgumentError AKLTQuickstart.run_demo(CONFIG, existing_file)
        @test read(existing_file, String) == "preserve me"
        @test isempty(filter(name -> startswith(name, ".aklt-demo-"), readdir(tmp)))
    end
end

@testset "Reject invalid configuration and options" begin
    mktempdir() do tmp
        base = TOML.parsefile(CONFIG)
        for (key, value) in (("L", 8), ("L", 11), ("L", 100), ("L", true),
            ("p_nnn", -0.1), ("p_nnn", 1.1), ("p_nnn", NaN), ("p_nnn", Inf),
            ("seed", -1), ("seed", 2.5), ("sample_every", 0), ("sample_every", 3),
            ("sample_start", 25), ("sweeps", 101), ("unexpected", 1))
            invalid = merge(base, Dict(key => value))
            file = joinpath(tmp, "invalid.toml")
            open(io -> TOML.print(io, invalid), file, "w")
            @test_throws ArgumentError AKLTQuickstart.run_demo(file, joinpath(tmp, "must_not_exist"))
            @test !ispath(joinpath(tmp, "must_not_exist"))
        end
        missing = copy(base)
        delete!(missing, "seed")
        file = joinpath(tmp, "missing.toml")
        open(io -> TOML.print(io, missing), file, "w")
        @test_throws ArgumentError AKLTQuickstart.read_config(file)
    end
    @test_throws ArgumentError AKLTQuickstart.main(["--unknown", "x"])
    @test_throws ArgumentError AKLTQuickstart.main(["--output"])
    @test_throws ArgumentError AKLTQuickstart.main(["--output", "a", "--output", "b"])
    @test_throws ArgumentError AKLTQuickstart.main(["--output", "--config"])
end
