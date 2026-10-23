using Test, Random, LinearAlgebra, DelimitedFiles

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "simulations/statevector/order/AKLTStatevectorOrderOnly.jl"))
include(joinpath(ROOT, "simulations/statevector/relaxation/AKLTStatevectorNNNZ.jl"))
module StaticRun
include(joinpath(@__DIR__, "../simulations/statevector/static/AKLTStatevector.jl"))
end
module RenyiRun
include(joinpath(@__DIR__, "../simulations/statevector/renyi/run_sv_nnn_alpha_worker.jl"))
end

# Independent dense embedding, deliberately separate from the optimized kernels.
function dense_pair(P, L, i, j)
    result = zeros(ComplexF64, 3^L, 3^L)
    for column in 0:3^L-1
        basis = digits(column; base=3, pad=L)
        pair_column = 3 * basis[i] + basis[j] + 1
        for a in 0:2, b in 0:2
            target = copy(basis)
            target[i], target[j] = a, b
            row = 1 + sum(target[k] * 3^(k-1) for k in 1:L)
            result[row, column+1] = P[3*a+b+1, pair_column]
        end
    end
    result
end

@testset "Projectors and normalized forced updates" begin
    for engine in (AKLTStatevectorOrderOnly, AKLTStatevectorNNNZ,
                   StaticRun.AKLTStatevector, RenyiRun.AKLTStatevector)
        P = engine.PNOT2
        @test P' == P
        @test P * P ≈ P atol=1e-12
        @test real(tr(P)) ≈ 4
        psi0 = normalize(randn(MersenneTwister(42), ComplexF64, 3^4))
        for (i,j) in ((1,2), (1,3), (4,1))
            psi = copy(psi0)
            reference = normalize(dense_pair(P, 4, i, j) * psi0)
            engine.project_normalize!(psi, i, j, 4, zeros(ComplexF64,9), zeros(ComplexF64,9))
            @test norm(psi) ≈ 1 atol=1e-12
            @test psi ≈ reference atol=1e-12
        end
    end
end

@testset "Seeded order-only output" begin
    mktempdir() do tmp
        first = joinpath(tmp, "first")
        second = joinpath(tmp, "second")
        for prefix in (first, second)
            # Both NNN separations exist only when L/2 - 1 >= 4.
            AKLTStatevectorOrderOnly.runy(10, 0.0, :periodic, 0.54, 4, 1, 5, 2, 4, 1, 1234, prefix)
        end
        files = filter(p -> startswith(basename(p), "first") && filesize(p) > 0, readdir(tmp; join=true))
        @test length(files) >= 4
        for file in files
            peer = joinpath(tmp, replace(basename(file), "first" => "second"; count=1))
            @test read(file) == read(peer)
            values = readdlm(file, ',', Float64)
            @test all(isfinite, values[2:end,2:end])
        end
    end
end

@testset "Renyi entropy conventions" begin
    for n in (0.7, 1.0, 2.0, 3.0, 4.0, 5.0, Inf)
        @test RenyiRun.entropy_from_probabilities_nats([0.5,0.5], n) ≈ log(2)
        @test RenyiRun.entropy_from_probabilities_nats([1.0,0.0], n) ≈ 0 atol=1e-12
    end
end

@testset "Relaxation and Renyi trajectories" begin
    mktempdir() do tmp
        AKLTStatevectorNNNZ.runy(6, 0.0, :periodic, 0.54, 12, 1, 6,
            6, 12, 6, 1234, joinpath(tmp,"entropy"), :entropy_only, "random", 5678, 2)
        files = readdir(tmp; join=true)
        entropy_files = filter(p -> endswith(p, "_S_SV_nnn_z"), files)
        @test length(entropy_files) == 1
        if length(entropy_files) == 1
            values = readdlm(only(entropy_files), ',', Float64)
            @test size(values) == (3,3)
            @test all(isfinite, values[2:end,2:end])
        end
    end
    task = Dict("L"=>"6", "p_nn"=>"0.54", "p_nnn"=>"0.46",
                "ti"=>"2", "tf"=>"3", "interval"=>"1", "bc"=>"periodic",
                "cut_mode"=>"standard", "self_av_cuts"=>"1", "campaign"=>"smoke")
    io = IOBuffer()
    specs = [RenyiRun.RenyiSpec("1",1.0), RenyiRun.RenyiSpec("2",2.0)]
    RenyiRun.run_trajectory!(io, task, 1, 1, 1, 1234, specs, [1e-12])
    rows = readdlm(IOBuffer(String(take!(io))), ',')
    @test size(rows,1) == 4
    @test all(isfinite, Float64.(rows[:,20:21]))
    @test Float64.(rows[:,20]) .* log(2) ≈ Float64.(rows[:,21])
end
