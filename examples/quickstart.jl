module AKLTQuickstart

using DelimitedFiles, SHA, Statistics, TOML

const ENGINE_PATH = joinpath(@__DIR__, "../simulations/statevector/order/AKLTStatevectorOrderOnly.jl")
include(ENGINE_PATH)
const ENGINE = AKLTStatevectorOrderOnly
const CONFIG_KEYS = Set(["L", "p_nnn", "seed", "sweeps", "sample_start", "sample_every"])
const CHANNELS = ("NN", "NNN", "NNm1", "NNNm1")
const SUMMARY_HEADER = ["sweep", "mean_nn", "mean_nnn", "abs_mean_nn", "abs_mean_nnn"]

"""Read and validate the deliberately small, periodic-chain demonstration."""
function read_config(path::AbstractString)
    config = TOML.parsefile(path)
    Set(keys(config)) == CONFIG_KEYS || throw(ArgumentError("Config requires exactly: $(join(sort!(collect(CONFIG_KEYS)), ", "))"))
    for key in ("L", "seed", "sweeps", "sample_start", "sample_every")
        value = config[key]
        value isa Integer && !(value isa Bool) || throw(ArgumentError("$key must be an integer"))
    end
    L = config["L"]
    iseven(L) && 10 <= L <= 12 || throw(ArgumentError("Demo L must be 10 or 12; larger runs use the campaign interfaces"))
    p = config["p_nnn"]
    p isa Real && !(p isa Bool) && isfinite(p) && 0 <= p <= 1 || throw(ArgumentError("p_nnn must be finite and in [0, 1]"))
    0 <= config["seed"] <= typemax(Int32) || throw(ArgumentError("seed must be in 0:$(typemax(Int32))"))
    1 <= config["sample_start"] <= config["sweeps"] <= 100 || throw(ArgumentError("Require 1 <= sample_start <= sweeps <= 100"))
    interval = config["sample_every"]
    interval > 0 || throw(ArgumentError("sample_every must be positive"))
    (config["sweeps"] - config["sample_start"]) % interval == 0 || throw(ArgumentError("Sampling interval must reach the final sweep exactly"))
    return config
end

function read_channel(rawdir, channel, config)
    suffix = "_SO_$(channel)_SV_nnn"
    files = filter(p -> endswith(p, suffix), readdir(rawdir; join=true))
    length(files) == 1 || error("Expected exactly one output for $channel; found $(length(files))")
    values = readdlm(only(files), ',', Float64)
    times = collect(config["sample_start"]:config["sample_every"]:config["sweeps"])
    size(values) == (length(times) + 1, div(config["L"], 2) + 1) || error("Unexpected $channel output shape")
    values[2:end, 1] == times || error("Incorrect sweep labels in $channel")
    all(==(config["seed"]), values[1, 2:end]) || error("Incorrect seed metadata in $channel")
    data = values[2:end, 2:end]
    all(isfinite, data) || error("Non-finite $channel observable")
    all(x -> abs(x) <= 1 + 1e-12, data) || error("$channel observable outside [-1, 1]")
    return data
end

"""Run the archived engine, validate its files, and publish a new output directory.

Returns its absolute path. Existing output paths are never replaced. The demo
records one trajectory, not a converged ensemble or a critical-exponent fit.
"""
function run_demo(config_path::AbstractString, output::AbstractString)
    config = read_config(config_path)
    destination = abspath(output)
    (ispath(destination) || islink(destination)) && throw(ArgumentError("Output already exists: $destination. Choose a new --output path."))
    parent = dirname(destination)
    mkpath(parent)
    stage = mktempdir(parent; prefix=".aklt-demo-")
    try
        rawdir = joinpath(stage, "raw")
        mkpath(rawdir)
        L = config["L"]
        ENGINE.runy(L, 0.0, :periodic, 1.0 - config["p_nnn"], config["sweeps"],
            1, div(L, 2), config["sample_start"], config["sweeps"],
            config["sample_every"], config["seed"], joinpath(rawdir, "trajectory"))
        channels = Dict(channel => read_channel(rawdir, channel, config) for channel in CHANNELS)
        times = collect(config["sample_start"]:config["sample_every"]:config["sweeps"])
        open(joinpath(stage, "samples.csv"), "w") do io
            println(io, "sweep,origin,seed,nn,nnn,nn_m1,nnn_m1")
            for (row, sweep) in enumerate(times), origin in 1:div(L, 2)
                writedlm(io, permutedims([sweep, origin, config["seed"],
                    (channels[key][row, origin] for key in CHANNELS)...]), ',')
            end
        end
        nn = vec(mean(channels["NN"]; dims=2))
        nnn = vec(mean(channels["NNN"]; dims=2))
        open(joinpath(stage, "summary.csv"), "w") do io
            println(io, join(SUMMARY_HEADER, ','))
            writedlm(io, hcat(times, nn, nnn, abs.(nn), abs.(nnn)), ',')
        end
        metadata = Dict(
            "parameters" => config,
            "julia_version" => string(VERSION),
            "engine" => "simulations/statevector/order/AKLTStatevectorOrderOnly.jl",
            "engine_sha256" => bytes2hex(open(sha256, ENGINE_PATH)),
            "p_nn" => 1.0 - config["p_nnn"],
            "boundary" => "periodic", "trajectories" => 1,
            "initial_state" => "uniform product state",
            "time_unit" => "one sweep = L local projection slots",
            "summary_convention" => "signed spatial mean, then absolute value; one trajectory only",
        )
        open(joinpath(stage, "metadata.toml"), "w") do io
            TOML.print(io, metadata; sorted=true)
        end
        mv(stage, destination)
        return destination
    finally
        isdir(stage) && rm(stage; recursive=true)
    end
end

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("Usage: julia --project=examples examples/quickstart.jl [--config FILE] [--output NEW_DIRECTORY]")
        return
    end
    options = Dict("--config" => joinpath(@__DIR__, "quickstart.toml"), "--output" => joinpath("results", "quickstart"))
    seen = Set{String}()
    iseven(length(args)) || throw(ArgumentError("Every option requires a value; use --help"))
    for i in 1:2:length(args)
        key, value = args[i], args[i+1]
        haskey(options, key) || throw(ArgumentError("Unknown option: $key"))
        key in seen && throw(ArgumentError("Repeated option: $key"))
        startswith(value, "--") && throw(ArgumentError("Missing value for $key"))
        push!(seen, key)
        options[key] = value
    end
    destination = run_demo(options["--config"], options["--output"])
    println("Validated outputs: $destination")
    println("One seeded trajectory; see summary.csv, samples.csv, metadata.toml, and raw/.")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        AKLTQuickstart.main()
    catch exception
        println(stderr, "Quickstart failed: ", sprint(showerror, exception))
        exit(1)
    end
end
